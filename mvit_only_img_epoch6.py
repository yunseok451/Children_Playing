auto_scale_lr = dict(base_batch_size=1024)
data_preprocessor = dict(num_classes=15, to_rgb=True)
dataset_type = 'ImageNet'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'epoch_6.bin'
log_level = 'INFO'
model = dict(
    backbone=dict(
        arch='tiny',
        drop_path_rate=0.3,
        init_cfg=dict(
            checkpoint='epoch_6.bin',
            prefix='backbone.',
            type='Pretrained'),
        type='MViT'),
    data_preprocessor=dict(num_classes=15, to_rgb=True),
    head=dict(
        in_channels=768,
        loss=dict(
            label_smooth_val=0.1, mode='original', type='LabelSmoothLoss'),
        num_classes=15,
        type='LinearClsHead'),
    init_cfg=[
        dict(bias=0.0, layer='Linear', std=0.02, type='TruncNormal'),
        dict(bias=0.0, layer='LayerNorm', type='Constant', val=1.0),
    ],
    neck=dict(type='GlobalAveragePooling'),
    train_cfg=dict(augments=[
        dict(alpha=0.8, type='Mixup'),
        dict(alpha=1.0, type='CutMix'),
    ]),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.00025,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
            '.rel_pos_h': dict(decay_mult=0.0),
            '.rel_pos_w': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
        flat_decay_mult=0.0,
        norm_decay_mult=0.0))
param_scheduler = [
    dict(
        by_epoch=True,
        convert_to_iter_based=True,
        end=70,
        start_factor=0.001,
        type='LinearLR'),
    dict(begin=70, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]
# randomness = dict( terministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                800,
                600,
            ), type='Resize'),
            dict(type='PackInputs'),
        ],
        type='ImageNet'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(topk=(1, ), type='Accuracy'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        800,
        600,
    ), type='Resize'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=11,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                800,
                600,
            ), type='Resize'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type='ImageNet'),
    num_workers=24,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        800,
        600,
    ), type='Resize'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=13,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                800,
                600,
            ), type='Resize'),
            dict(type='PackInputs'),
        ],
        type='ImageNet'),
    num_workers=24,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='ConfusionMatrix'),
    dict(topk=(1, ), type='Accuracy'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/workspace/mmpretrain/only_img_epoch6'
