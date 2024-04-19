import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags,
                        images_to_levels, multi_apply, reduce_mean, unmap)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import ATSSHead

EPSILON = 1e-5

@HEADS.register_module()
class ATSSKDQHead(ATSSHead):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 pred_kernel_size=3,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 reg_decoded_bbox=True,
                 centerness=0,
                 gt_mask=None,
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
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
                         name='atss_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        super(ATSSKDQHead, self).__init__(
            num_classes,
            in_channels,
            pred_kernel_size,
            stacked_convs,
            conv_cfg,
            norm_cfg,
            reg_decoded_bbox,
            loss_centerness,
            init_cfg,
            **kwargs)
        self.loss_kd = build_loss(loss_kd)
        self.centerness = centerness
        self.gt_mask = gt_mask
        if gt_mask is not None:
            self.alpha = gt_mask['alpha']

    def loss_single_w_kd(self, anchors, cls_score, bbox_pred, centerness, 
                    cls_score_tea, bbox_pred_tea, centerness_tea, 
                    labels, label_weights, bbox_targets, num_total_samples, gt_mask=None):
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
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        input_size = cls_score.size()

        resp = torch.cat([cls_score,
                          bbox_pred,
                          centerness], dim=1)
        resp = resp.permute(0, 2, 3, 1).reshape(-1, self.num_classes+4+1)

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        
        if input_size == cls_score_tea.size():
            resp_tea = torch.cat([cls_score_tea,
                            bbox_pred_tea,
                            centerness_tea], dim=1)
            resp_tea = resp_tea.permute(0, 2, 3, 1).reshape(-1, self.num_classes+4+1)
            # # knowledge distillation loss

            cls_score_tea = cls_score_tea.permute(0, 2, 3, 1).reshape(
                -1, self.cls_out_channels).contiguous()
            bbox_pred_tea = bbox_pred_tea.permute(0, 2, 3, 1).reshape(-1, 4)
            centerness_tea = centerness_tea.permute(0, 2, 3, 1).reshape(-1)

            if self.gt_mask is not None:
                gt_mask = gt_mask.unsqueeze(1)
                n = gt_mask.shape[-1] // input_size[-1]
                for _ in range(math.floor(math.log2(n))):
                    gt_mask = F.max_pool2d(gt_mask, 2, 2, ceil_mode=True)

                flatten_gt_mask = gt_mask.permute(0, 2, 3, 1).reshape(-1, 1)

                loss_kd = self.loss_kd(resp, resp_tea) * flatten_gt_mask
            else:
                loss_kd = self.loss_kd(cls_score, cls_score_tea) + \
                    self.loss_kd(bbox_pred, bbox_pred_tea) + \
                    self.loss_kd(centerness, centerness_tea)
        else:
            loss_kd = bbox_pred.sum() * 0

        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        
        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)
            
            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                centerness_targets,
                avg_factor=num_total_samples)
            
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)
        
        return loss_cls, loss_bbox, loss_centerness, loss_kd, centerness_targets.sum()


    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             cls_scores_tea=None,
             bbox_preds_tea=None,
             centernesses_tea=None,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

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
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)
        
        if cls_scores_tea is not None:
            for _ in range(self.s_gap):
                cls_scores_tea.append(torch.tensor(0, dtype=torch.float, device=device))
                bbox_preds_tea.append(torch.tensor(0, dtype=torch.float, device=device))
                centernesses_tea.append(torch.tensor(0, dtype=torch.float, device=device))
            
            if self.gt_mask is not None:
                with torch.no_grad():
                    gt_mask = self.create_elipsis_mask(gt_bboxes, self.gt_mask['img_shape'])
            else:
                gt_mask = None

            losses_cls, losses_bbox, loss_centerness, losses_kd, \
                bbox_avg_factor = multi_apply(
                    self.loss_single_w_kd,
                    anchor_list,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    cls_scores_tea[self.s_gap:],
                    bbox_preds_tea[self.s_gap:],
                    centernesses_tea[self.s_gap:],
                    labels_list,
                    label_weights_list,
                    bbox_targets_list,
                    num_total_samples=num_total_samples,
                    gt_mask=gt_mask)

            bbox_avg_factor = sum(bbox_avg_factor)
            bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
            losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
            return dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox,
                loss_centerness=loss_centerness,
                losses_resp_kd=losses_kd)
        else:
            losses_cls, losses_bbox, loss_centerness, \
                bbox_avg_factor = multi_apply(
                    self.loss_single,
                    anchor_list,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    labels_list,
                    label_weights_list,
                    bbox_targets_list,
                    num_total_samples=num_total_samples)
            
            bbox_avg_factor = sum(bbox_avg_factor)
            bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
            losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
            return dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox,
                loss_centerness=loss_centerness)

    def forward_train(self,
                      x,
                      out_teacher,
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
            losses = self.loss(*loss_inputs, *out_teacher, gt_bboxes_ignore=gt_bboxes_ignore)
        else:
            losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list
        
    def centerness_target(self, anchors, gts):
        # only calculate pos centerness targets, otherwise there may be nan
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        if self.centerness == 0:
            centerness = torch.sqrt(
                (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
                (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        elif self.centerness == 1:
            r = (top_bottom.sum(dim=-1) + left_right.sum(dim=-1)) / 2
            centerness = (left_right.min(dim=-1)[0] + top_bottom.min(dim=-1)[0]) / r
        centerness = torch.clamp(centerness, min=0, max=1)
        assert not torch.isnan(centerness).any()
        return centerness

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    cls_scores=None,
                    bbox_preds=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        if cls_scores is None:
            cls_scores = [None for _ in range(num_imgs)]
        if bbox_preds is None:
            bbox_preds = [None for _ in range(num_imgs)]
        
        num_levels = len(cls_scores)
        cls_score_list = []
        bbox_pred_list = []

        mlvl_cls_score_list = [
            cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        mlvl_bbox_pred_list = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        for i in range(num_imgs):
            mlvl_cls_tensor_list = [
                mlvl_cls_score_list[j][i] for j in range(num_levels)
            ]
            mlvl_bbox_tensor_list = [
                mlvl_bbox_pred_list[j][i] for j in range(num_levels)
            ]
            cat_mlvl_cls_score = torch.cat(mlvl_cls_tensor_list, dim=0)
            cat_mlvl_bbox_pred = torch.cat(mlvl_bbox_tensor_list, dim=0)
            cls_score_list.append(cat_mlvl_cls_score)
            bbox_pred_list.append(cat_mlvl_bbox_pred)


        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             cls_score_list,
             bbox_pred_list,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           cls_scores,
                           bbox_preds,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            cls_scores (Tensor): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W)
            bbox_preds (Tensor): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W)
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        
        dec_bbox_preds = self.bbox_coder.decode(anchors, bbox_preds)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels,
                                             cls_scores=cls_scores,
                                             bbox_preds=dec_bbox_preds)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)
        
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if self.reg_decoded_bbox:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

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