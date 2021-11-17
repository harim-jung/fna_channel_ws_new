# model settings
type = 'YOLOX'
model = dict(
    type='NASYOLOX', # NASYolox
    # pretrained=dict(
    #     use_load=True,
    #     load_path='./seed_mbv2.pt',        
    #     seed_num_layers=[1, 1, 2, 3, 4, 3, 3, 1, 1] # mbv2
    #     ),
    backbone=dict(
        type='YOLOXBackbone', # YoloxBackbone
        init_cfg=dict(
            type='Pretrained', 
            use_load=True, 
            # use_load=False, 
            checkpoint='./seed_mbv2.pt',
            seed_num_layers=[1, 1, 2, 3, 4, 3, 3, 1, 1]), # mbv2
        search_params=dict(
            sample_policy='prob', # prob uniform
            weight_sample_num=1,
            affine=False,
            track=False,
            net_scale=dict(
                # chs = [32, 16, 24, 32, 64, 96, 160, 320], # 고정된 layer를 추가해야 하는것인지
                chs = [32, 16, 24, 32, 64, 128, 256, 512],
                num_layers = [4, 4, 4, 4, 4, 1],
                strides = [2, 1, 2, 2, 2, 1, 2, 1, 1],
            ),
            primitives_normal=['k3_e3',
                                'k3_e6',
                                'k5_e3',
                                'k5_e6',
                                'k7_e3',
                                'k7_e6',
                                'skip',],
            primitives_reduce=['k3_e3',
                                'k3_e6',
                                'k5_e3',
                                'k5_e6',
                                'k7_e3',
                                'k7_e6',],
        ),
        output_indices=[5, 6, 7],

    ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512], # s
        # in_channels=[192, 384, 768], # m
        # in_channels=[96, 160, 320],
        # out_channels=192, # m
        out_channels=128, # s
        # out_channels=96,
        # num_csp_blocks=2, # m
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', 
        num_classes=80, 
        # in_channels=192, # m
        in_channels=128, 
        # feat_channels=192, # m
        feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../datasets/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (640, 640)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=114.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=img_scale, pad_val=114.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=3,
    # samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        # type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(
    weight_optim = dict(
        optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
        optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    ),
    arch_optim = dict(
        optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.001,
                        betas=(0.5, 0.999)),
        optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    )
)
# optimizer = dict(
#     type='SGD',
#     lr=0.01,
#     momentum=0.9,
#     weight_decay=5e-4,
#     nesterov=True,
#     paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
# optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11, 14])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    # interval=500,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# configs for sub_obj optimizing
sub_obj=dict(
    if_sub_obj=True,
    type='flops',
    log_base=10.,
    sub_loss_factor=0.02
)
# yapf:enable
# runtime settings
total_epochs = 14

use_syncbn = False

arch_update_epoch = 8
alter_type = 'step' # step / epoch
train_data_ratio = 0.5
image_size_madds = (800, 1088)
model_info_interval = 1000
device_ids = range(8)
dist_params = dict(backend='nccl')
# log_level = 'DEBUG'
log_level = 'INFO'
work_dir = './work_dirs/'
load_from = None
resume_from = None
workflow = [('arch', 1),('train', 1)]

# yapf:enable
# custom_hooks = [dict(type='NumClassCheckHook')]
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(
        type='SyncRandomSizeHook',
        ratio_range=(14, 26),
        img_scale=img_scale,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=15,
        interval=1,
        priority=48),
    dict(type='ExpMomentumEMAHook', resume_from=resume_from, priority=49)
]