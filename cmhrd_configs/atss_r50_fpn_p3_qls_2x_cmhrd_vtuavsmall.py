_base_ = [
    '../configs/_base_/schedules/schedule_2x.py', '../configs/_base_/default_runtime.py'
]
model = dict(
    type='ATSSHRKD',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSKDQHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        centerness=1,
        gt_mask=dict(alpha=0., img_shape=(270,480,3)),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_kd=dict(type='L1Loss', loss_weight=0.05) # no mask: 0.05; mask: 5.0
        ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='QLSAssigner', 
                      topk=9,
                      alpha=0.8,
                      quality='x',
                      iou_calculator=dict(type='BboxDistanceMetric'),
                      iou_mode='siwd',
                      overlap_mode='hybrid',
                      ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100),
    teacher_cfg='cmhrd_configs/atss_r50_fpn_p3_qls_1x_vtuavsmall_v.py',
    teacher_ckpt='work_dirs/atss_r50_fpn_p3_qls/vtuavsmall/visible-hr/epoch_11_best.pth',
    feature_distill=dict(scale_gap=2,
                         highlevel=dict(
                            type='cross',
                            weight=0.05,
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
                                featmap_strides=[2, 4, 8, 16, 32],
                                finest_scale=224,
                                ),
                                         ),
                         ),
    response_distill=dict(),
    )


dataset_type = 'VTUAVdet_S'
data_root = '/home/zhangyan/data/VTUAV-det/'
img_norm_cfg = dict(
    mean_list=([97.17, 90.61, 79.91], [140.60, 140.60, 140.60]),
    std_list=([57.63, 57.07, 57.84], [83.15, 83.15, 83.15]), to_rgb=True)
train_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('visible', 'thermal')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MultiResize', img_scale=[(1920, 1080), (480, 270)], keep_ratio=True, main_scale_idx=2),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='MultiNormalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
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
            dict(type='Pad', size_divisor=32),
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
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)   # 0.005 for 4 gpus
optimizer_config = dict(grad_clip=dict(_delete_=True, max_norm=35, norm_type=2))


work_dir = 'work_dirs/atss_r50_fpn_p3_qls_cmhrd/vtuavsmall'
