model = dict(
    type='YOLOF',
    backbone=dict(
        type='FNA_Yolof',
        pretrained=dict(
            use_load=True,
            load_path='./seed_mbv2.pt',
            seed_num_layers=[1, 1, 2, 3, 4, 3, 3, 1, 1] # mbv2
        ),
# output/yolof_2048_adapt/net_config_epoch_14_6500.txt
#         net_config="""[[32, 16], ['k3_e1'], 1]|
# [[16, 24], ['k3_e3', 'skip', 'skip', 'skip'], 2]|
# [[24, 32], ['k3_e6', 'skip', 'k7_e3', 'k7_e3'], 2]|
# [[32, 64], ['k7_e3', 'k3_e3', 'k5_e3', 'k3_e3'], 2]|
# [[64, 96], ['k3_e3', 'skip', 'skip', 'skip'], 1]|
# [[96, 160], ['k7_e6', 'k5_e3', 'skip', 'skip'], 2]|
# [[160, 320], ['k3_e3'], 1]|
# [[320, 2048], ['conv_2d_1x1'], 1]""",
# output/yolof_1280_adapt/net_config_epoch_13_5000.txt
#         net_config="""[[32, 16], ['k3_e1'], 1]|
# [[16, 24], ['k5_e6', 'skip', 'skip', 'skip'], 2]|
# [[24, 32], ['k3_e6', 'k7_e6', 'skip', 'skip'], 2]|
# [[32, 64], ['k7_e6', 'k7_e6', 'k3_e3', 'k5_e6'], 2]|
# [[64, 96], ['k3_e6', 'skip', 'skip', 'k5_e3'], 1]|
# [[96, 160], ['k7_e3', 'skip', 'skip', 'k5_e3'], 2]|
# [[160, 320], ['k3_e3'], 1]|
# [[320, 1280], ['conv_2d_1x1'], 1]""",
# output/yolof_1280_sub_obj_0.1_adapt/net_config_epoch_14.txt
        net_config="""[[32, 16], ['k3_e1'], 1]|
[[16, 24], ['k5_e6', 'skip', 'skip', 'k3_e3'], 2]|
[[24, 32], ['k3_e6', 'k7_e6', 'skip', 'k7_e6'], 2]|
[[32, 64], ['k7_e6', 'k7_e6', 'k7_e6', 'k5_e6'], 2]|
[[64, 96], ['k5_e6', 'skip', 'skip', 'k7_e3'], 1]|
[[96, 160], ['k7_e6', 'k7_e3', 'k7_e3', 'k5_e3'], 2]|
[[160, 320], ['k3_e3'], 1]|
[[320, 1280], ['conv_2d_1x1'], 1]""",
        output_indices=[8],
    ),
    neck=dict(
        type='DilatedEncoder',
        # in_channels=2048,
        in_channels=1280,
        out_channels=512,
        block_mid_channels=128,
        num_residual_blocks=4),
    bbox_head=dict(
        type='YOLOFHead',
        num_classes=80,
        in_channels=512,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[1, 2, 4, 8, 16],
            strides=[32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1., 1., 1., 1.],
            add_ctr_clamp=True,
            ctr_clamp=32),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='UniformAssigner', pos_ignore_thr=0.15, neg_ignore_thr=0.7),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

dataset_type = 'CocoDataset'
data_root = '../../datasets/coco/'
# same as yolof search
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# yolof training default
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=False)
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
    samples_per_gpu=16, # 16 * 2 = 32
    workers_per_gpu=4,
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
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')


# optimizer
# optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.00005) # lr 0.05
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001,
                 paramwise_cfg=dict(norm_decay_mult=0., custom_keys={'backbone': dict(lr_mult=1. / 3)})) # lr 0.05
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    # warmup_iters=500,
    warmup_iters=1500,
    # warmup_ratio=1.0 / 3,
    warmup_ratio=0.00066667,
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
    imports=['models.derived_yolof_backbone'],
    allow_failed_imports=False)