# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from ..builder import HEADS, build_loss
from .retina_head import RetinaHead
from mmdet.core import images_to_levels, multi_apply

EPSILON = 1e-5

@HEADS.register_module()
class RetinaKDHead(RetinaHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 gt_mask=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 loss_kd=dict(
                     type='L1Loss',
                     loss_weight=1.0,
                     reduction='none',
                 ),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_kd = build_loss(loss_kd)
        self.gt_mask = gt_mask
        if gt_mask is not None:
            self.alpha = gt_mask['alpha']

    def forward_train(self,
                      x,
                      out_teacher,
                      teacher_uncertainty,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        if out_teacher is not None:
            if teacher_uncertainty is not None:
                losses = self.loss(*loss_inputs, *out_teacher, *teacher_uncertainty, gt_bboxes_ignore=gt_bboxes_ignore)
            else:
                losses = self.loss(*loss_inputs, *out_teacher, gt_bboxes_ignore=gt_bboxes_ignore)
        else:
            losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def loss_single_w_kd(self, cls_score, bbox_pred, cls_score_tea, bbox_pred_tea, 
                         anchors, labels, label_weights,
                         bbox_targets, bbox_weights, num_total_samples, gt_mask=None):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        input_size = cls_score.size()
        max_cls_score, max_index = torch.max(cls_score, 1, True)

        max_bbox_pred = bbox_pred.reshape(-1, 4)
        max_index = max_index.reshape(-1)
        max_bbox_pred = max_bbox_pred[max_index, :]
        max_bbox_pred = max_bbox_pred.reshape(input_size[0], 4, input_size[-2], input_size[-1])
        resp = torch.cat([max_cls_score, \
                        max_bbox_pred], dim=1)
        
        resp = resp.permute(0, 2, 3, 1).reshape(-1, 5)

        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        
        if self.gt_mask is not None:
            gt_mask = gt_mask.unsqueeze(1)
            n = gt_mask.shape[-1] // input_size[-1]
            for _ in range(math.floor(math.log2(n))):
                gt_mask = F.max_pool2d(gt_mask, 2, 2, ceil_mode=True)
            gt_mask = gt_mask.repeat(1, 5, 1, 1)
            flatten_gt_mask = gt_mask.permute(0, 2, 3, 1).reshape(-1, 5)

            max_cls_score_tea, max_index_tea = torch.max(cls_score_tea, 1, True)
            max_bbox_pred_tea = bbox_pred_tea.reshape(-1, 4)
            max_index_tea = max_index_tea.reshape(-1)
            max_bbox_pred_tea = max_bbox_pred_tea[max_index_tea, :]
            max_bbox_pred_tea = max_bbox_pred_tea.reshape(input_size[0], 4, input_size[-2], input_size[-1])
            resp_tea = torch.cat([max_cls_score_tea, \
                            max_bbox_pred_tea], dim=1)
            resp_tea = resp_tea.permute(0, 2, 3, 1).reshape(-1, 5)
            
            loss_kd = self.loss_kd(resp, resp_tea) * flatten_gt_mask
        else:
            loss_kd = self.loss_kd(cls_score, cls_score_tea) + \
                self.loss_kd(bbox_pred, bbox_pred_tea)

        return loss_cls, loss_bbox, loss_kd

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cls_scores_tea=None,
             bbox_preds_tea=None,
             cls_tea_uncertainty=None,
             bbox_tea_uncertainty=None,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        if cls_scores_tea is not None:
            for _ in range(self.s_gap):
                cls_scores_tea.append(torch.tensor(0, dtype=torch.float, device=device))
                bbox_preds_tea.append(torch.tensor(0, dtype=torch.float, device=device))
            if cls_tea_uncertainty is not None:
                for _ in range(self.s_gap):
                    cls_tea_uncertainty.append(torch.tensor(0, dtype=torch.float, device=device))
                    bbox_tea_uncertainty.append(torch.tensor(0, dtype=torch.float, device=device))
            else:
                cls_tea_uncertainty = [None for _ in range(len(cls_scores_tea))]
                bbox_tea_uncertainty = [None for _ in range(len(bbox_preds_tea))]
            
            if self.gt_mask is not None:
                with torch.no_grad():
                    gt_mask = self.create_elipsis_mask(gt_bboxes, self.gt_mask['img_shape'])
            else:
                gt_mask = None
            
            losses_cls, losses_bbox, losses_kd = multi_apply(
                self.loss_single_w_kd,
                cls_scores,
                bbox_preds,
                cls_scores_tea[self.s_gap:],
                bbox_preds_tea[self.s_gap:],                
                all_anchor_list,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                num_total_samples=num_total_samples,
                gt_mask=gt_mask)
            
            return dict(loss_cls=losses_cls, 
                        loss_bbox=losses_bbox,
                        loss_resp_kd=losses_kd,
                        )
        else:
            losses_cls, losses_bbox = multi_apply(
                self.loss_single,
                cls_scores,
                bbox_preds,
                all_anchor_list,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                num_total_samples=num_total_samples)
            return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def create_elipsis_mask(self, bboxes, img_shape=(512,640,3), mode=1):
        import numpy as np
        h, w, c = img_shape
        bs = len(bboxes)
        mask = torch.zeros((bs, h, w), device='cuda')
        
        for i, batch_bboxes in enumerate(bboxes):
            n = len(batch_bboxes)
            mask_ = torch.zeros((n, h, w))
            xmin, ymin, xmax, ymax = batch_bboxes[:, 0], batch_bboxes[:, 1], batch_bboxes[:, 2], batch_bboxes[:, 3]
            a = (xmax - xmin)/2
            b = (ymax - ymin)/2
            cx = (xmax + xmin)/2
            cy = (ymax + ymin)/2
            # import pdb; pdb.set_trace()
            a = (a - torch.maximum(cx, w-cx)) * self.alpha + torch.maximum(cx, w-cx)
            b = (b - torch.maximum(cy, h-cy)) * self.alpha + torch.maximum(cy, h-cy)

            x = torch.tensor(np.arange(0, w, 1), device='cuda:0')
            y = torch.tensor(np.arange(0, h, 1), device='cuda:0')
            xx, yy = torch.meshgrid(x, y, indexing='xy')

            for j in range(n):
                # binary mask
                if mode == 1:
                    m_ = (xx-cx[j])**2/a[j]**2 + (yy-cy[j])**2/b[j]**2 < 1+EPSILON
                    m_.to(dtype=torch.float32)
                    mask_[j, :, :] = m_
                else:
                    # gauss heatmap
                    gauss_map = gaus2d(xx, yy, cx[j], cy[j], a[j], b[j])
                    mask_[j, :, :] = gauss_map.clone().detach()


            mask[i, :, :] = mask_.sum(0).clamp_(min=0, max=1)

        return mask

def gaus2d(xx, yy, mux=0, muy=0, sigx=1, sigy=1, A=1):
    return A*torch.exp(-((xx - mux)**2. / (2. * sigx**2.) + (yy - muy)**2. / (2. * sigy**2.)))