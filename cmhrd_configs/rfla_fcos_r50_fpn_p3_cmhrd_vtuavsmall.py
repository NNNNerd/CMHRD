_base_ = [
    '../configs/_base_/schedules/schedule_2x.py', '../configs/_base_/default_runtime.py'
]
# model settings
model = dict(
    type='FCOSCMHRD',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RFLA_FCOSKDHead',
        norm_cfg=None,
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        conv_bias=True,
        fpn_layer = 'p3', # bottom FPN layer P2
        fraction = 1/2,
        gt_mask=dict(alpha=0.0, img_shape=(270,480,3)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='DIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_kd=dict(type='L1Loss', loss_weight=0.005, reduction='none') # no mask: 0.05; mask: 5.0
            ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HieAssigner',
             ignore_iof_thr=-1,
             gpu_assign_thr=256,
             iou_calculator=dict(type='BboxDistanceMetric'),
             assign_metric='kl',
             topk=[3,1],
             ratio=0.9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=3000),
    teacher_cfg='cmhrd_configs/rfla_fcos_r50_fpn_p3_vtuavsmall_v.py',
    teacher_ckpt='work_dirs/rfla_fcos/vtuavsmall/visible-hr/epoch_11_best.pth',
    feature_distill=dict(scale_gap=2,
                         highlevel=dict(
                            type='cross',
                            weight=0.006,
                            idx=[2,3,4]
                         ),
                         sr_feat_kd=dict(
                             weight=0.5,
                             spa_attn=False,
                             fuse=False,
                             bbox_roi_extractor_s=dict(
                                type='SingleRoIExtractor',
                                roi_layer=dict(type='RoIAlign', output_size=3, sampling_ratio=0),
                                out_channels=256,
                                featmap_strides=[8, 16, 32, 64, 128],
                                finest_scale=56,
                                ),
                             bbox_roi_extractor_t=dict(
                                type='SingleRoIExtractor',
                                roi_layer=dict(type='RoIAlign', output_size=12, sampling_ratio=0),
                                out_channels=256,
                                featmap_strides=[8, 16, 32, 64, 128],
                                finest_scale=224,
                                ),
                                         ),
    
    ),
    response_distill=dict(),
)

# dataset settings
dataset_type = 'VTUAVdet_S'
data_root = '/home/zhangyan/data/VTUAV-det/'
img_norm_cfg = dict(
    mean_list=([110.47, 112.07, 110.47], [140.60, 140.60, 140.60]),
    std_list=([97.17, 90.61, 79.91], [57.63, 57.07, 57.84]), to_rgb=True)
train_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MultiResize', img_scale=[(1920, 1080), (480, 270)], keep_ratio=True, main_scale_idx=2),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='MultiNormalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(480, 270),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='MultiNormalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'small_train_thermal.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'small_test_thermal.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'small_test_thermal.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(
    lr=0.02, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(_delete_=True, max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=10000,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

work_dir = 'work_dirs/rfla_fcos_cmhrd/noaa'