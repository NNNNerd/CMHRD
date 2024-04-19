# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_backbone, build_head
from .retinanet import RetinaNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from torch.cuda.amp import autocast

from mmcv.ops import batched_nms
from mmcv.runner import BaseModule, force_fp32
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from mmdet.core import bbox2result


@DETECTORS.register_module()
class QueryDet(RetinaNet):
    """Implementation of `an RGBT fusion version of QueryDet 
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
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(QueryDet, self).__init__(backbone, neck, bbox_head, train_cfg,
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

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        query_features = [x[i] for i in self.query_layers]
        outs = self.bbox_head(x)
        query_outs = self.query_head(query_features)
        return (outs, query_outs)
    
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        
        query_features = [x[i] for i in self.query_layers]
        # det loss
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                        gt_labels, gt_bboxes_ignore)
        # query loss
        query_loss = self.query_head.forward_train(query_features, img_metas, gt_bboxes,
                                        gt_labels, gt_bboxes_ignore)
        losses.update(query_loss)
        # weight = torch.linspace(1, 2.6, steps=len(x), device=losses['loss_cls'][0].device)
        # for k in losses.keys():
        #     for i in range(len(losses[k])):
        #         losses[k][i] = weight[i]*losses[k][i]
        return losses
    
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
