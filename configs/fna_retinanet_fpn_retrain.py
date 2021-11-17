# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='FNA_Retinanet',
        pretrained=dict(
            use_load=True,
            load_path='./seed_mbv2.pt',
            seed_num_layers=[1, 1, 2, 3, 4, 3, 3, 1, 1] # mbv2
        ),
        # net config for alpha test
        net_config="""[[32, 16], ['k3_e1'], 1]|
[[16, 24], ['k5_e6', 'skip', 'skip', 'k3_e6'], 2]|
[[24, 32], ['k7_e6', 'skip', 'k3_e6', 'k3_e6'], 2]|
[[32, 64], ['k7_e6', 'k5_e6', 'k3_e6', 'k3_e6'], 2]|
[[64, 96], ['k3_e6', 'skip', 'skip', 'k7_e6'], 1]|
[[96, 160], ['k7_e6', 'k7_e6', 'k7_e6', 'k7_e6'], 2]|
[[160, 320], ['k7_e6'], 1]""",
        output_indices=[2, 3, 5, 7]
    ),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
            # from fna
            smoothl1_beta=0.11,
            gamma=2.0,
            alpha=0.25,
            # from mmdet
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))


# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../datasets/coco/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
        dict(type='Resize', keep_ratio=True),
        dict(type='RandomFlip'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img']),
    ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_test2017/image_info_test-dev2017.json',
        img_prefix=data_root + 'test2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.00005) # lr 0.05
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
        ]
    )
custom_hooks = [dict(type='NumClassCheckHook')]
# runner = dict(type='EpochBasedRunner', max_epochs=12)
runner = dict(type='EpochBasedRunner', max_epochs=15)

# yapf:enable
# runtime settings
# total_epochs = 12
use_syncbn = True
image_size_madds = (800, 1088)
device_ids = range(8)

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/retinanet_FNA_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]


custom_imports = dict(
    imports=['models.derived_retinanet_backbone'],
    allow_failed_imports=False)