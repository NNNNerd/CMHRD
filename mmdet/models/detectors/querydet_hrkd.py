# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_head, build_detector, build_roi_extractor
from .retinanet import RetinaNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from torch.cuda.amp import autocast

import mmcv
from mmcv.runner import load_checkpoint
from mmcv.ops import batched_nms
from mmcv.runner import BaseModule, force_fp32
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from mmdet.core import bbox2result


@DETECTORS.register_module()
class QueryDetHRKD(RetinaNet):
    """Implementation of QueryDet 
        <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 query_head,
                 query_layers=[1,2],
                 whole_feat_id_test=[2,3,4,5],
                 value_feat_id_test=[0,1],
                 query_infer=False,
                 query_threshold=0.15,
                 query_context=2,
                 feature_distill=None,
                 response_distill=None,
                 reconstruct_cfg=None,
                 reweight=None,
                 train_cfg=None,
                 test_cfg=None,
                 teacher_cfg=None,
                 eval_teacher=True,
                 teacher_ckpt=None,
                 pretrained=None,
                 init_cfg=None):
        super(QueryDetHRKD, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)
        self.query_head = build_head(query_head)
        self.query_layers = query_layers
        self.whole_feat_id_test = whole_feat_id_test
        self.value_feat_id_test = value_feat_id_test
        self.query_threshold = query_threshold
        self.query_context = query_context
        self.qInfer = QueryInfer(9, self.bbox_head.num_classes, self.query_threshold, self.query_context)
        self.query_infer = query_infer
        self.iter = 0

        if isinstance(teacher_cfg, str):
            teacher_cfg = mmcv.Config.fromfile(teacher_cfg)
        self.teacher_model = build_detector(teacher_cfg['model'])
        self.eval_teacher = eval_teacher
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')
        self.cuda(device=torch.device("cuda"))
        
        self.feature_distill = feature_distill
        self.response_distill = response_distill
        self.reconstruct_cfg = reconstruct_cfg

        self.s_gap = self.feature_distill['scale_gap']
        self.bbox_head.s_gap = self.feature_distill['scale_gap']

        if 'sr_feat_kd' in feature_distill:
            self.sr_cfg = feature_distill['sr_feat_kd']
            self.sr_generation_layer = FeatureSRModule(neck['out_channels'], 2 ** self.s_gap)
            self.bbox_roi_extractor_s = build_roi_extractor(self.sr_cfg['bbox_roi_extractor_s'])
            self.bbox_roi_extractor_t = build_roi_extractor(self.sr_cfg['bbox_roi_extractor_t'])
        else:
            self.sr_cfg = None
        
        if 'lowlevel' in feature_distill:
            self.low_cfg = feature_distill['lowlevel']
            self.adp_layer_low = nn.Conv2d(neck['out_channels'], neck['out_channels'], 1, 1)
            if self.low_cfg['deform']:
                self.offset_layer = nn.Sequential(nn.Conv2d(neck['out_channels'], 2, 1, 1),
                                          nn.Tanh())
            if self.low_cfg['type'] =='mgd':
                self.generation_layer = nn.Sequential(
                    nn.Conv2d(neck['out_channels'], neck['out_channels'], kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(neck['out_channels'], neck['out_channels'], kernel_size=3, padding=1)
            )
        else:
            self.low_cfg = None
        
        if 'highlevel' in feature_distill:
            self.high_cfg = feature_distill['highlevel']
            self.adp_layer_high = nn.Conv2d(neck['out_channels'], neck['out_channels'], 1, 1)
        else:
            self.high_cfg = None
        
        if self.reconstruct_cfg is not None:
            self.bbox_roi_extractor = build_roi_extractor(reconstruct_cfg['bbox_roi_extractor'])
            if self.reconstruct_cfg['with_rgb']:
                self.reconstructor = Reconstructor(reconstruct_cfg['bbox_roi_extractor']['out_channels'], 2)
            else:
                self.reconstructor = Reconstructor(reconstruct_cfg['bbox_roi_extractor']['out_channels'], 1)
        
        self.criterion = nn.L1Loss(reduction='none')
        self.recons_loss = nn.MSELoss(reduction='mean')

        self.reweight = reweight

        self.iteration = 0


    def forward_dummy(self, img):
        x = self.extract_feat(img)
        query_features = [x[i] for i in self.query_layers]
        outs = self.bbox_head(x)
        query_outs = self.query_head(query_features)
        return (outs, query_outs)
    
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        x = self.extract_feat(img[1])

        losses = dict()

        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img[0])

            teacher_x = list(teacher_x)
            for _ in range(self.s_gap):
                teacher_x.append(F.max_pool2d(teacher_x[-1], 1, stride=2))
            
            if self.response_distill is not None:
                out_teacher = self.teacher_model.bbox_head(teacher_x)
                uncertainty_teacher = None
        
        if self.sr_cfg is not None:
            sr_gen_loss = self.feat_sr_gen_kd_loss(x, teacher_x, gt_bboxes)
            losses.update(dict(sr_gen_loss=sr_gen_loss))

        if self.low_cfg is not None:
            if self.low_cfg['type'] == 'mask':
                feat_kd_low_loss = [self.low_cfg['weight'] * self.feat_kd_loss(self.adp_layer_low(x[i]), 
                                                                                teacher_x[i+self.s_gap], 
                                                                                gt_bboxes, 
                                                                                img_metas[0]['img_shape']) 
                                                                                for i in self.low_cfg['idx']]
            elif self.low_cfg['type'] == 'pts':
                feat_kd_low_loss = [self.low_cfg['weight'] * self.feat_pts_kd_loss(self.adp_layer_low(x[i]), 
                                                                                    teacher_x[i], 
                                                                                    gt_bboxes, 
                                                                                    img_metas[0]['img_shape']) 
                                                                                    for i in self.low_cfg['idx']]
            elif self.low_cfg['type'] == 'mgd':
                feat_kd_low_loss = [self.low_cfg['weight'] * self.feat_obj_mgd_loss(self.adp_layer_low(x[i]), 
                                                                                    teacher_x[i+self.s_gap], 
                                                                                    gt_bboxes, 
                                                                                    img_metas[0]['img_shape']) 
                                                                                    for i in self.low_cfg['idx']]
            else:
                raise NotImplementedError(self.low_cfg['type'])
            losses.update(dict(feat_kd_low_loss=feat_kd_low_loss))
        
        if self.high_cfg is not None:
            feat_kd_high_loss = [self.high_cfg['weight'] * self.feat_kd_high_loss(self.adp_layer_high(x[i]), 
                                                                                    teacher_x[i+self.s_gap]) 
                                                                                    for i in self.high_cfg['idx']]
            losses.update(dict(feat_kd_high_loss=feat_kd_high_loss))
        

        query_features = [x[i] for i in self.query_layers]
        # det loss
        if self.response_distill is not None:
            bbox_losses = self.bbox_head.forward_train(x, out_teacher, uncertainty_teacher, img_metas,
                                                gt_bboxes, gt_labels,
                                                gt_bboxes_ignore)
        else:
            bbox_losses = self.bbox_head.forward_train(x, None, None, img_metas,
                                                gt_bboxes, gt_labels,
                                                gt_bboxes_ignore)
        losses.update(bbox_losses)

        # query loss
        query_loss = self.query_head.forward_train(query_features, img_metas, gt_bboxes,
                                        gt_labels, gt_bboxes_ignore)
        losses.update(query_loss)

        if self.reweight is not None:
            weight = torch.linspace(self.reweight['w_min'], self.reweight['w_max'], steps=len(x), device=losses['loss_cls'][0].device)
            for k in losses.keys():
                for i in range(len(losses[k])):
                    losses[k][i] = weight[i]*losses[k][i]
        return losses
 
    def feat_kd_loss(self, x, teacher_x, gt_bboxes, img_shape, mask_mode='gaussian'):
        import math
        assert x.shape == teacher_x.shape
        masks = create_gt_mask(gt_bboxes, img_shape, mode=mask_mode)
        masks = masks.unsqueeze(1)
        n = masks.shape[-1] // x.shape[-1]
        # import pdb; pdb.set_trace()
        for _ in range(math.floor(math.log2(n))):
            masks = F.max_pool2d(masks, 2, 2, ceil_mode=True)
        # x = x * masks
        # teacher_x = teacher_x * masks
        # act = torch.softmax(x, dim=1)
        # act_tea = torch.softmax(teacher_x, dim=1)

        loss = self.criterion(x, teacher_x) * masks        
        
        if self.feature_distill['attn_weight']:
            attn_weight = spatial_attention_weight(teacher_x)
            loss = loss.reshape(x.shape[0], x.shape[1], -1) * attn_weight

        loss = torch.mean(loss)
        return loss
    
    def feat_pts_kd_loss(self, feat, teacher_feat, gt_bboxes, img_shape):
        H, W, _ = img_shape
        bs, ch, h, w = feat.shape
        scale = (h / H, w/ W)

        if self.low_cfg['deform']:
            offset_maps = self.offset_layer(feat)

        sample_feats = []
        tea_sample_feats = []
        for i, gts in enumerate(gt_bboxes):
            n_boxes = gts.size(0)
            mapped_gts = torch.clone(gts)
            mapped_gts[:, 0] = mapped_gts[:, 0]/W *2 -1
            mapped_gts[:, 1] = mapped_gts[:, 1]/H *2 -1
            mapped_gts[:, 2] = mapped_gts[:, 2]/W *2 -1
            mapped_gts[:, 3] = mapped_gts[:, 3]/H *2 -1
            h_b = mapped_gts[:, 3] - mapped_gts[:, 1]
            w_b = mapped_gts[:, 2] - mapped_gts[:, 0]
            cx = (mapped_gts[:, 0] + mapped_gts[:, 2]) / 2.0
            cy = (mapped_gts[:, 1] + mapped_gts[:, 3]) / 2.0
            c = torch.stack((cx, cy), dim=1)
            v1 = mapped_gts[:, 0:2]
            v2 = torch.stack((mapped_gts[:, 1], mapped_gts[:, 2]), dim=1)
            v3 = torch.stack((mapped_gts[:, 0], mapped_gts[:, 3]), dim=1)
            v4 = mapped_gts[:, 2:]
            e1 = (v1 + v2) / 2.0
            e2 = (v1 + v3) / 2.0
            e3 = (v2 + v4) / 2.0
            e4 = (v3 + v4) / 2.0
            sample_points = torch.stack([v1, e1, v2, \
                                          e2, c, e3, \
                                          v3, e4, v4], dim=1)
            sample_points = torch.reshape(sample_points, (n_boxes, 3, 3, 2))

            feat_per_batch = feat[i:i+1, :, :, :]
            teacher_feat_per_batch = teacher_feat[i:i+1, :, :, :]
            feat_per_batch = feat_per_batch.expand(n_boxes, -1, -1, -1)
            teacher_feat_per_batch = teacher_feat_per_batch.expand(n_boxes, -1, -1, -1)
            
            if self.low_cfg['deform']:
                offset_map = offset_maps[i:i+1, :, :, :]
                offset_map = offset_map.expand(n_boxes, -1, -1, -1)
                sample_offsets = F.grid_sample(offset_map, sample_points)
                sample_offsets = torch.permute(sample_offsets, (0, 2, 3, 1))
                offset_bnd = torch.stack([w_b*0.5, h_b*0.5], dim=1)
                offset_bnd = offset_bnd.unsqueeze(1).expand(-1,9,-1).reshape(-1,3,3,2)
                sample_offsets = sample_offsets * offset_bnd

                sample_points = torch.clamp(sample_points + sample_offsets, min=-1, max=1)

            sample_feat = F.grid_sample(feat_per_batch, sample_points)
            tea_sample_feat = F.grid_sample(teacher_feat_per_batch, sample_points)

            if self.low_cfg['aff']:
                flatten_sample_feat = sample_feat.reshape(n_boxes, ch, -1)
                flatten_tea_sample_feat = tea_sample_feat.reshape(n_boxes, ch, -1)
                stu_aff = torch.bmm(flatten_sample_feat.permute(0, 2, 1), flatten_sample_feat)
                tea_aff = torch.bmm(flatten_tea_sample_feat.permute(0, 2, 1), flatten_sample_feat)
                
                sample_feat_norm = torch.linalg.vector_norm(flatten_sample_feat, dim=1, ord=2)
                tea_sample_feat_norm = torch.linalg.vector_norm(flatten_tea_sample_feat, dim=1, ord=2)
                stu_weight = torch.bmm(sample_feat_norm.unsqueeze(-1),sample_feat_norm.unsqueeze(1))
                tea_weight = torch.bmm(tea_sample_feat_norm.unsqueeze(-1),sample_feat_norm.unsqueeze(1))
                stu_aff = stu_aff / stu_weight
                tea_aff = tea_aff / tea_weight
                
                sample_feats.append(stu_aff)
                tea_sample_feats.append(tea_aff)
            else:
                sample_feats.append(sample_feat)
                tea_sample_feats.append(tea_sample_feat)
            
        sample_feats = torch.cat(sample_feats, dim=0)
        tea_sample_feats = torch.cat(tea_sample_feats, dim=0)
        
        return self.criterion(sample_feats, tea_sample_feats)

    def feat_obj_mgd_loss(self, feat, teacher_feat, gt_bboxes, img_shape):
        bs, c, h, w = feat.shape
        import math
        masks = create_gt_mask(gt_bboxes, img_shape, mode='elipsis')
        masks = masks.unsqueeze(1)
        n = masks.shape[-1] // feat.shape[-1]
        for _ in range(math.floor(math.log2(n))):
            masks = F.max_pool2d(masks, 2, 2, ceil_mode=True)
        
        device = feat.device
        mat = torch.rand((bs,c,h,w)).to(device)
        mat = torch.where(mat>1-self.low_cfg['mgd_prob'], 0, 1).to(device)

        masked_feat = feat * mat
        new_feat = self.generation_layer(masked_feat)

        loss = self.criterion(new_feat, teacher_feat) * masks
        if self.feature_distill['attn_weight']:
            attn_weight = spatial_attention_weight(teacher_feat)
            loss = loss.reshape(feat.shape[0], feat.shape[1], -1) * attn_weight

        loss = torch.mean(loss)
        return loss

    def feat_sr_gen_kd_loss(self, feats, teacher_feats, gt_bboxes):
        # multi-scale object
        # feature super-resolution generative distillation
        batch_ids_list = []
        for i, gts in enumerate(gt_bboxes):
            batch_ids = gts.new_full((gts.size(0), 1), i)
            batch_ids_list.append(batch_ids)
        batch_ids_list = torch.cat(batch_ids_list, dim=0)
        flatten_gt_regions = torch.cat(gt_bboxes, dim=0)
        # flatten_tea_gt_regions = flatten_gt_regions * (2**self.s_gap)

        flatten_gt_regions = torch.cat([batch_ids_list, flatten_gt_regions], dim=1)
        # flatten_tea_gt_regions = torch.cat([batch_ids_list, flatten_tea_gt_regions], dim=1)

        # spatial attention fused teacher feat
        if self.sr_cfg['fuse']:
            up_feats = []
            for i in range(4):
                up_feats.append(size_alignment(feats[i], teacher_feats[i].shape[-2:]))
            teacher_feats = [spatial_fusion(up_feats[i], teacher_feats[i]).detach() for i in range(4)]

        gt_feat = self.bbox_roi_extractor_s(feats, flatten_gt_regions)
        tea_gt_feat = self.bbox_roi_extractor_t(teacher_feats, flatten_gt_regions)

        ## ablation study
        # gen_gt_feat = gt_feat
        gen_gt_feat = self.sr_generation_layer(gt_feat)

        # spatial attention weighted loss
        if self.sr_cfg['spa_attn']:
            spa_w = [spatial_attention_weight(teacher_feat) for teacher_feat in teacher_feats]
            sampled_spa_w = self.bbox_roi_extractor_t(spa_w, flatten_gt_regions)

            loss = self.criterion(gen_gt_feat, tea_gt_feat) * sampled_spa_w
        else:
            loss = self.criterion(gen_gt_feat, tea_gt_feat)

        loss = torch.mean(loss) * self.sr_cfg['weight']

        return loss


    def feat_kd_high_loss(self, feat, teacher_feat):
        bs, c, h, w = feat.shape
        flatten_feat = feat.reshape(bs, c, -1)
        flatten_teacher_feat = teacher_feat.reshape(bs, c, -1)

        stu_aff = torch.bmm(flatten_feat.permute(0, 2, 1), flatten_feat)
        if self.high_cfg['type'] == 'cross':
            tea_aff = torch.bmm(flatten_teacher_feat.permute(0, 2, 1), flatten_feat).detach()
        else:
            tea_aff = torch.bmm(flatten_teacher_feat.permute(0, 2, 1), flatten_teacher_feat)
        # if self.high_cfg['norm']:
        #     sample_feat_norm = torch.linalg.vector_norm(flatten_feat, dim=1, ord=2)
        #     tea_sample_feat_norm = torch.linalg.vector_norm(flatten_teacher_feat, dim=1, ord=2)
        #     stu_weight = torch.bmm(sample_feat_norm.unsqueeze(-1),sample_feat_norm.unsqueeze(1))
        #     tea_weight = torch.bmm(tea_sample_feat_norm.unsqueeze(-1),sample_feat_norm.unsqueeze(1))
        #     stu_aff = stu_aff / stu_weight
        #     tea_aff = tea_aff / tea_weight


        loss = self.criterion(stu_aff, tea_aff).mean()
        
        return loss
    
    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img[1])
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
    # def simple_test(self, img, img_metas, rescale=False):
    #     self.iter += 1

    #     feat = self.extract_feat(img)
    #     whole_feat = [feat[i] for i in self.whole_feat_id_test]
    #     value_feat = [feat[i] for i in self.value_feat_id_test]
    #     key_feat = [feat[i] for i in self.query_layers]
    #     # import pdb; pdb.set_trace()
    #     save_feature_to_img(feat, 'features', self.iter)

    #     featmap_sizes = [f.size()[-2:] for f in feat]
    #     multi_level_anchors = self.bbox_head.prior_generator.grid_priors(
    #         featmap_sizes, device="cuda")
    #     import pdb; pdb.set_trace()
    #     anchors_whole  = [multi_level_anchors[i] for i in self.whole_feat_id_test]  
    #     anchors_value  = [multi_level_anchors[i] for i in self.value_feat_id_test]

    #     det_cls_whole, det_delta_whole = self.bbox_head(whole_feat)
        
    #     if not self.query_infer:
    #         det_cls_query, det_bbox_query = self.bbox_head(value_feat)
    #         det_cls_query = [permute_to_N_HWA_K(x, self.bbox_head.num_classes) for x in det_cls_query]
    #         det_bbox_query = [permute_to_N_HWA_K(x, 4) for x in det_bbox_query]
    #         query_anchors = anchors_value
    #     else:
    #         if not self.qInfer.initialized:
    #             cls_weights, cls_biases, bbox_weights, bbox_biases = self.bbox_head.get_params()
    #             qcls_weights, qcls_bias = self.query_head.get_params()
    #             params = [cls_weights, cls_biases, bbox_weights, bbox_biases, qcls_weights, qcls_bias]
    #         else:
    #             params = None

    #         det_cls_query, det_bbox_query, query_anchors = self.qInfer.run_qinfer(params, key_feat, value_feat, anchors_value)

    #     cls_scores, bbox_preds, anchors = self.merge(det_cls_whole, det_delta_whole, anchors_whole,\
    #                 det_cls_query, det_bbox_query, query_anchors, img_metas=img_metas)
        
    #     results_list = self.get_bboxes(cls_scores, bbox_preds, anchors, img_metas=img_metas, cfg=self.test_cfg)

    #     bbox_results = [
    #         bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #         for det_bboxes, det_labels in results_list
    #     ]
    #     return bbox_results
    
    def merge(self, retina_box_cls, retina_box_delta, retina_anchors,
                    small_det_logits, small_det_delta, small_det_anchors, img_metas=None):
        N, _, _, _ = retina_box_cls[0].size()
        retina_box_cls = [permute_to_N_HWA_K(x, self.bbox_head.num_classes) for x in retina_box_cls]
        retina_box_delta = [permute_to_N_HWA_K(x, 4) for x in retina_box_delta]
        
        if self.query_infer:
            small_det_logits = [small_det_logits[i].view(N, -1, self.bbox_head.num_classes) for i in range(len(small_det_logits)-1,-1,-1)]
            small_det_delta = [small_det_delta[i].view(N, -1, 4) for i in range(len(small_det_delta)-1,-1,-1)]
            small_det_anchors = [small_det_anchors[i] for i in range(len(small_det_anchors)-1,-1,-1)]
        else:
            small_det_logits = [x.view(N, -1, self.bbox_head.num_classes) for x in small_det_logits]
            small_det_delta = [x.view(N, -1, 4) for x in small_det_delta]

        for img_idx, img_meta in enumerate(img_metas):
            
            retina_box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in retina_box_cls]
            retina_box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in retina_box_delta]
            small_det_logits_per_image = [small_det_cls_per_level[img_idx] for small_det_cls_per_level in small_det_logits]
            small_det_reg_per_image = [small_det_reg_per_level[img_idx] for small_det_reg_per_level in small_det_delta]
            
            if len(small_det_anchors) == 0 or type(small_det_anchors[0]) == torch.Tensor:
                small_det_anchor_per_image = [small_det_anchor_per_level[img_idx] for small_det_anchor_per_level in small_det_anchors]
            else:
                small_det_anchor_per_image = small_det_anchors
        
        all_cls = small_det_logits + retina_box_cls
        all_delta = small_det_delta + retina_box_delta 
        all_anchors = small_det_anchors + retina_anchors
        
        return all_cls, all_delta, all_anchors

    def get_bboxes(self, cls_scores, bbox_preds, mlvl_anchors,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        # featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, mlvl_anchors,
                                              img_meta, cfg, rescale, with_nms)
            result_list.append(results)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):
            assert cls_score.size()[0] == bbox_pred.size()[0]

            # bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            # cls_score = cls_score.permute(1, 2,
                                        #   0).reshape(-1, self.bbox_head.cls_out_channels)
            if self.bbox_head.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            # import pdb; pdb.set_trace()
            # assert bbox_pred.size() == priors.size()
            results = filter_scores_and_topk(scores, cfg.score_thr, nms_pre, dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_head.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)
            
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):

        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                return det_bboxes, mlvl_labels

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                                mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels


    def inference(self, 
                  retina_box_cls, retina_box_delta, retina_anchors,
                  small_det_logits, small_det_delta, small_det_anchors, 
                  img_metas
    ):
        results = []

        N, _, _, _ = retina_box_cls[0].size()
        retina_box_cls = [permute_to_N_HWA_K(x, self.bbox_head.num_classes) for x in retina_box_cls]
        retina_box_delta = [permute_to_N_HWA_K(x, 4) for x in retina_box_delta]
        small_det_logits = [x.view(N, -1, self.bbox_head.num_classes) for x in small_det_logits]
        small_det_delta = [x.view(N, -1, 4) for x in small_det_delta]
        # import pdb; pdb.set_trace()

        for img_idx, img_meta in enumerate(img_metas):
            
            retina_box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in retina_box_cls]
            retina_box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in retina_box_delta]
            small_det_logits_per_image = [small_det_cls_per_level[img_idx] for small_det_cls_per_level in small_det_logits]
            small_det_reg_per_image = [small_det_reg_per_level[img_idx] for small_det_reg_per_level in small_det_delta]
            
            if len(small_det_anchors) == 0 or type(small_det_anchors[0]) == torch.Tensor:
                small_det_anchor_per_image = [small_det_anchor_per_level[img_idx] for small_det_anchor_per_level in small_det_anchors]
            else:
                small_det_anchor_per_image = small_det_anchors
     
            results_per_img = self.inference_single_image(
                                retina_box_cls_per_image, retina_box_reg_per_image, retina_anchors,
                                small_det_logits_per_image, small_det_reg_per_image, small_det_anchor_per_image,
                                img_meta['img_shape'])
            results.append(results_per_img)

        return results


    def inference_single_image(self, 
                               retina_box_cls, retina_box_delta, retina_anchors, 
                               small_det_logits, small_det_delta, small_det_anchors, 
                               image_size
    ):  
        with autocast(False):
            # small pos cls inference
            all_cls = small_det_logits + retina_box_cls
            all_delta = small_det_delta + retina_box_delta 
            all_anchors = small_det_anchors + retina_anchors

            boxes_all, scores_all, class_idxs_all = self.decode_dets(all_cls, all_delta, all_anchors)
            boxes_all, scores_all, class_idxs_all = [cat(x) for x in [boxes_all, scores_all, class_idxs_all]]
            
            if self.use_soft_nms:  
                keep, soft_nms_scores = self.soft_nmser(boxes_all, scores_all, class_idxs_all)
            else:
                keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
            result = Instances(image_size)

            keep = keep[: self.max_detections_per_image]       
            result.pred_boxes = Boxes(boxes_all[keep])
            result.scores = scores_all[keep]
            result.pred_classes = class_idxs_all[keep]
            return result


class QueryInfer(object):
    def __init__(self, anchor_num, num_classes, score_th=0.12, context=2):
        
        self.anchor_num  = anchor_num
        self.num_classes = num_classes
        self.score_th    = score_th
        self.context     = context 

        self.initialized = False
        self.cls_spconv  = None 
        self.bbox_spconv = None
        self.qcls_spconv = None
        self.qcls_conv   = None 
        self.n_conv      = None
    
    
    def _make_sparse_tensor(self, query_logits, last_ys, last_xs, anchors, feature_value):
        if last_ys is None:
            N, _, qh, qw = query_logits.size()
            assert N == 1
            prob  = torch.sigmoid_(query_logits).view(-1)
            pidxs = torch.where(prob > self.score_th)[0]# .float()
            y = torch.div(pidxs, qw).int()
            x = torch.remainder(pidxs, qw).int()
        else:
            prob  = torch.sigmoid_(query_logits).view(-1)
            pidxs = prob > self.score_th
            y = last_ys[pidxs]
            x = last_xs[pidxs]
        
        if y.size(0) == 0:
            return None, None, None, None, None, None 

        _, fc, fh, fw = feature_value.shape
        
        ys, xs = [], []
        for i in range(2):
            for j in range(2):
                ys.append(y * 2 + i)
                xs.append(x * 2 + j)

        ys = torch.cat(ys, dim=0)
        xs = torch.cat(xs, dim=0)
        inds = (ys * fw + xs).long()

        sparse_ys = []
        sparse_xs = []
        
        for i in range(-1*self.context, self.context+1):
            for j in range(-1*self.context, self.context+1):
                sparse_ys.append(ys+i)
                sparse_xs.append(xs+j)

        sparse_ys = torch.cat(sparse_ys, dim=0)
        sparse_xs = torch.cat(sparse_xs, dim=0)


        good_idx = (sparse_ys >= 0) & (sparse_ys < fh) & (sparse_xs >= 0)  & (sparse_xs < fw)
        sparse_ys = sparse_ys[good_idx]
        sparse_xs = sparse_xs[good_idx]
        
        sparse_yx = torch.stack((sparse_ys, sparse_xs), dim=0).t()
        sparse_yx = torch.unique(sparse_yx, sorted=False, dim=0)
        
        sparse_ys = sparse_yx[:, 0]
        sparse_xs = sparse_yx[:, 1]

        sparse_inds = (sparse_ys * fw + sparse_xs).long()

        sparse_features = feature_value.view(fc, -1).transpose(0, 1)[sparse_inds].view(-1, fc)
        sparse_indices  = torch.stack((torch.zeros_like(sparse_ys), sparse_ys, sparse_xs), dim=-1)  
        sparse_tensor = spconv.SparseConvTensor(sparse_features, sparse_indices.int(), (fh, fw), 1)
  
        anchors = anchors.view(-1, self.anchor_num, 4)
        selected_anchors = anchors[inds].view(1, -1, 4)
        return sparse_tensor, ys, xs, inds, selected_anchors, sparse_indices.size(0)

    def _make_spconv(self, weights, biases):
        nets = []
        for i in range(len(weights)):
            in_channel  = weights[i].shape[1]
            out_channel = weights[i].shape[0]
            k_size      = weights[i].shape[2]
            filter = spconv.SubMConv2d(in_channel, out_channel, k_size, 1, padding=k_size//2, indice_key="asd", algo=spconv.ConvAlgo.Native).to(device=weights[i].device)
            # filter.weight.data[:] = weights[i].permute(2,3,1,0).contiguous()[:] # transpose(1,2).transpose(0,1).transpose(2,3).transpose(1,2).transpose(2,3)
            # import pdb; pdb.set_trace()
            filter.weight.data[:] = weights[i].permute(0,2,3,1).contiguous()[:]
            filter.bias.data   = biases[i]
            nets.append(filter)
            if i != len(weights) - 1:
                nets.append(torch.nn.ReLU(inplace=True))
        return spconv.SparseSequential(*nets)

    def _make_conv(self, weights, biases):
        nets = []
        for i in range(len(weights)):
            in_channel  = weights[i].shape[0]
            out_channel = weights[i].shape[1]
            k_size      = weights[i].shape[2]
            filter = torch.nn.Conv2d(in_channel, out_channel, k_size, 1, padding=k_size//2)
            filter.weight.data = weights[i]
            filter.bias.data   = biases[i]
            nets.append(filter)
            if i != len(weights) - 1:
                nets.append(torch.nn.ReLU())
        return torch.nn.Sequential(*nets)
    
    def _run_spconvs(self, x, filters):
        y = filters(x)
        return y.dense(channels_first=False)

    def _run_convs(self, x, filters):
        return filters(x)

    def run_qinfer(self, model_params, features_key, features_value, anchors_value):
        
        if not self.initialized:
            cls_weights, cls_biases, bbox_weights, bbox_biases, qcls_weights, qcls_biases = model_params
            assert len(cls_weights) == len(qcls_weights)
            self.n_conv = len(cls_weights)
            self.cls_spconv  = self._make_spconv(cls_weights, cls_biases)
            self.bbox_spconv = self._make_spconv(bbox_weights, bbox_biases)
            self.qcls_spconv = self._make_spconv(qcls_weights, qcls_biases)
            self.qcls_conv   = self._make_conv(qcls_weights, qcls_biases)
            self.initialized  = True

        last_ys, last_xs = None, None 
        query_logits = self._run_convs(features_key[-1], self.qcls_conv)
        det_cls_query, det_bbox_query, query_anchors = [], [], []
        
        n_inds_all = []

        for i in range(len(features_value)-1, -1, -1):
            x, last_ys, last_xs, inds, selected_anchors, n_inds = self._make_sparse_tensor(query_logits, last_ys, last_xs, anchors_value[i], features_value[i])
            n_inds_all.append(n_inds)
            if x == None:
                break
            cls_result   = self._run_spconvs(x, self.cls_spconv).view(-1, self.anchor_num*self.num_classes)[inds]
            bbox_result  = self._run_spconvs(x, self.bbox_spconv).view(-1, self.anchor_num*4)[inds]
            query_logits = self._run_spconvs(x, self.qcls_spconv).view(-1)[inds]
            
            query_anchors.append(selected_anchors.squeeze())
            det_cls_query.append(torch.unsqueeze(cls_result, 0))
            det_bbox_query.append(torch.unsqueeze(bbox_result, 0))

        return det_cls_query, det_bbox_query, query_anchors


def permute_to_N_HWA_K(tensor, K):
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor

def run_conv2d(x, weights, bias):
    n_conv = len(weights)
    for i in range(n_conv):
        x = F.conv2d(x, weights[i], bias[i])
        if i != n_conv - 1:
            x = F.relu(x)
    return x


class FeatureSRModule(nn.Module):
    def __init__(self, in_channels, upscale_factor, method='learnable'):
        super(FeatureSRModule, self).__init__()
        self.method = method
        # self.sr_size = sr_size
        self.upscale_factor = upscale_factor
        self.res_block = ResidualBlock(in_channels)
        self.upsample_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
        )

    def forward(self, x):
        if self.method == 'bicubic':
            x = F.interpolate(x, scale_factor=self.upscale_factor)

        elif self.method == 'learnable':
            x = self.res_block(x)
            x = self.upsample_block(x)
        else:
            raise NotImplementedError(self.method)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
		    nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.up(x)
        return x


class Reconstructor(nn.Module):
    def __init__(self, in_channles, output_ch=1):
        super().__init__()
        
        self.up4 = UpConv(in_channels=in_channles, out_channels=in_channles)
        self.up_conv4 = ConvBlock(in_channels=in_channles, out_channels=in_channles)
        self.up3 = UpConv(in_channels=in_channles, out_channels=in_channles)
        self.up_conv3 = ConvBlock(in_channels=in_channles, out_channels=in_channles)
        self.up2 = UpConv(in_channels=in_channles, out_channels=in_channles)
        self.up_conv2 = ConvBlock(in_channels=in_channles, out_channels=in_channles)

        self.conv1x1 = nn.Conv2d(in_channles, output_ch, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        d4 = self.up4(x)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = self.up_conv2(d2)

        d1 = self.conv1x1(d2)

        return d1


EPSILON = 1e-5

def create_gt_mask(bboxes, img_shape, mode='rec'):
    h, w, c = img_shape
    bs = len(bboxes)
    mask = torch.zeros((bs, h, w), device='cuda')
    for i, batch_bboxes in enumerate(bboxes):
        n = len(batch_bboxes)
        mask_ = torch.zeros((n, h, w))
        xmin, ymin, xmax, ymax = batch_bboxes[:, 0], batch_bboxes[:, 1], batch_bboxes[:, 2], batch_bboxes[:, 3]

        x = torch.tensor(np.arange(0, w, 1), device='cuda:0')
        y = torch.tensor(np.arange(0, h, 1), device='cuda:0')
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        if mode == 'rec':
            for j in range(n):
                m_ = (xx - xmax[j]) * (xx - xmin[j]) < EPSILON and (yy - ymax[j]) * (yy - ymin[j]) < EPSILON
                m_.to(dtype=torch.float32)
                mask_[j, :, :] = m_
        else:
            a = (xmax - xmin)/2
            b = (ymax - ymin)/2
            cx = (xmax + xmin)/2
            cy = (ymax + ymin)/2

            for j in range(n):
                # binary mask
                if mode == 'elipsis':
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

def size_alignment(x, size):
    if x.shape[-2:] == size:
        return x
    else:
        return F.interpolate(x, size=size)

def spatial_attention_weight(x, type='sigmoid'):
    shape = x.size()
    spatial_weight = x.mean(dim=1, keepdim=True)
    if type == 'softmax':
        spatial_weight = F.softmax(spatial_weight.reshape(shape[0], 1, -1), dim=-1) * 50
    elif type == 'sigmoid':
        spatial_weight = F.sigmoid(spatial_weight.reshape(shape[0], 1, -1))
    spatial_weight = spatial_weight.repeat(1, shape[1], 1)
    return spatial_weight.reshape(shape[0], shape[1], shape[-1], shape[-2])

def spatial_fusion(x1, x2):
    shape = x1.size()
    # calculate spatial attention
    spatial1 = x1.mean(dim=1, keepdim=True)
    spatial2 = x2.mean(dim=1, keepdim=True)
    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
    tensor_f = spatial_w1 * x1 + spatial_w2 * x2
    return tensor_f

def get_gt_image(img, gt_samples, size, with_rgb=True):
    from torchvision.transforms.functional import resize
    scale_gap = img[0].shape[-1] // img[1].shape[-1]
    gt_images = []
    if gt_samples.size(0) > 0:
        for i in range(gt_samples.size(0)):
            batch_id = gt_samples[i, 0].long()
            x1, y1, x2, y2 = gt_samples[i, 1:].long()
            if with_rgb:
                img_crop_v = img[0][batch_id:batch_id+1, :, y1*scale_gap:y2*scale_gap, x1*scale_gap:x2*scale_gap]
                img_crop_t = F.interpolate(img[1][batch_id:batch_id+1, :, y1:y2, x1:x2], scale_factor=scale_gap)
                img_crop_v = img_crop_v.mean(1, keepdim=True)
                img_crop_t = img_crop_t.mean(1, keepdim=True)
                img_crop = torch.cat([img_crop_t, img_crop_v], dim=1)
            else:
                img_crop = img[1][batch_id:batch_id+1, :, y1:y2, x1:x2]
                img_crop = img_crop.mean(1, keepdim=True)
            img_crop = resize(img_crop, size)
            gt_images.append(img_crop)
        gt_images = torch.cat(gt_images, dim=0)
    else:
        gt_images = img[0].new_zeros((0, 2, size[0], size[1]))
    return gt_images
