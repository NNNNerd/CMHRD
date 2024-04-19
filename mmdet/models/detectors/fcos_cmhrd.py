# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage_cmhrd import CMHRD


@DETECTORS.register_module()
class FCOSCMHRD(CMHRD):
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
        super(FCOSCMHRD, self).__init__(backbone, 
                                       neck, 
                                       bbox_head, 
                                       feature_distill,
                                       response_distill,
                                       train_cfg,
                                       test_cfg, 
                                       teacher_cfg,
                                       eval_teacher,
                                       teacher_ckpt,
                                       pretrained, 
                                       init_cfg)

