# model settings
type = 'Retinanet'
model = dict(
    type='NASRetinaNet', 
    backbone=dict(
        type='RetinaNetBackbone',
        pretrained=dict(
            use_load=True,
            load_path='./seed_mbv2.pt',        
            seed_num_layers=[1, 1, 2, 3, 4, 3, 3, 1, 1] # mbv2
        ),
        # init_cfg=dict(
        #     type='Pretrained', 
        #     use_load=True, 
        #     # use_load=False, 
        #     checkpoint='./seed_mbv2.pt',
        #     seed_num_layers=[1, 1, 2, 3, 4, 3, 3, 1, 1]), # mbv2
        search_params=dict(
            sample_policy='prob', # prob uniform
            weight_sample_num=1,
            affine=False,
            track=False,
            net_scale=dict(
                chs = [32, 16, 24, 32, 64, 96, 160, 320],
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
        output_indices=[2, 3, 5, 7],

    ),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
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
    
    # training and testing settings
    train_cfg = dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        smoothl1_beta=0.11,
        gamma=2.0,
        alpha=0.25,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100))
            
# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../datasets/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
    samples_per_gpu=2,
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
    # interval=10,
    interval=500,
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