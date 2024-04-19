import torch
import torch.nn as nn
import torch.nn.functional as F

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_detector, build_roi_extractor
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class CMHRD(SingleStageDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 feature_distill=None,
                 response_distill=None,
                 train_cfg=None,
                 test_cfg=None,
                 teacher_cfg=None,
                 eval_teacher=True,
                 teacher_ckpt=None,
                 pretrained=None,
                 init_cfg=None):
        super(CMHRD, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
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

        self.s_gap = self.feature_distill['scale_gap']
        self.bbox_head.s_gap = self.feature_distill['scale_gap']

        if 'sr_feat_kd' in feature_distill:
            self.sr_cfg = feature_distill['sr_feat_kd']
            self.sr_generation_layer = FeatureSRModule(neck['out_channels'], 2 ** self.s_gap)
            self.bbox_roi_extractor_s = build_roi_extractor(self.sr_cfg['bbox_roi_extractor_s'])
            self.bbox_roi_extractor_t = build_roi_extractor(self.sr_cfg['bbox_roi_extractor_t'])
        else:
            self.sr_cfg = None
                
        if 'highlevel' in feature_distill:
            self.high_cfg = feature_distill['highlevel']
            self.adp_layer_high = nn.Conv2d(neck['out_channels'], neck['out_channels'], 1, 1)
        else:
            self.high_cfg = None
                
        self.criterion = nn.L1Loss(reduction='none')
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = dict()
        
        x = self.extract_feat(img[1])
        self.img_scale = img[0].shape[-1] / img[1].shape[-1]
        
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img[0])
        
            teacher_x = list(teacher_x)
            teacher_x.append(F.max_pool2d(teacher_x[-1], 1, stride=2))
            teacher_x.append(F.max_pool2d(teacher_x[-1], 1, stride=2))
            out_teacher = self.teacher_model.bbox_head(teacher_x)

        # SRG Loss
        if self.sr_cfg is not None:
            sr_gen_loss = self.feat_sr_gen_kd_loss(x, teacher_x, gt_bboxes)
            losses.update(dict(sr_gen_loss=sr_gen_loss))

        # CMA Loss
        if self.high_cfg is not None:
            feat_kd_high_loss = [self.high_cfg['weight'] * self.feat_kd_high_loss(self.adp_layer_high(x[i]),
                                                                                    teacher_x[i+self.s_gap]) 
                                                                                    for i in self.high_cfg['idx']]
            losses.update(dict(feat_kd_high_loss=feat_kd_high_loss))
        
        # Response KD
        if self.response_distill is not None:
            bbox_losses = self.bbox_head.forward_train(x, out_teacher, img_metas,
                                                gt_bboxes, gt_labels,
                                                gt_bboxes_ignore)
        else:
            bbox_losses = self.bbox_head.forward_train(x, None, None, img_metas,
                                                gt_bboxes, gt_labels,
                                                gt_bboxes_ignore)            
        
        losses.update(bbox_losses)
        return losses
    
    def feat_sr_gen_kd_loss(self, feats, teacher_feats, gt_bboxes):
        # multi-scale object
        # feature super-resolution generative distillation
        batch_ids_list = []
        for i, gts in enumerate(gt_bboxes):
            batch_ids = gts.new_full((gts.size(0), 1), i)
            batch_ids_list.append(batch_ids)
        batch_ids_list = torch.cat(batch_ids_list, dim=0)
        flatten_gt_regions = torch.cat(gt_bboxes, dim=0)
        flatten_tea_gt_regions = flatten_gt_regions * self.img_scale

        flatten_gt_regions = torch.cat([batch_ids_list, flatten_gt_regions], dim=1)
        flatten_tea_gt_regions = torch.cat([batch_ids_list, flatten_tea_gt_regions], dim=1)

        gt_feat = self.bbox_roi_extractor_s(feats, flatten_gt_regions)
        tea_gt_feat = self.bbox_roi_extractor_t(teacher_feats, flatten_tea_gt_regions)

        gen_gt_feat = self.sr_generation_layer(gt_feat)

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

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)


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


