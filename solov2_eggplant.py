_base_ = 'solov2_r50_fpn_1x_coco.py'

# =========================
# Dataset
# =========================
dataset_type = 'CocoDataset'
data_root = 'C:/Users/gelob/Desktop/mmdetection format dataset/'

classes = ('infested', 'non-infested')

# =========================
# Model
# =========================
model = dict(
    backbone=dict(
        frozen_stages=1,
        norm_eval=True
    ),
    mask_head=dict(
        num_classes=2
    )
)

# =========================
# Pipelines
# Light pipeline only because Roboflow already applied augmentation
# =========================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')
]

# =========================
# Dataloaders
# =========================
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=True, min_size=16)
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        pipeline=test_pipeline
    )
)

# =========================
# Evaluators
# =========================
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric=['bbox', 'segm']
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/_annotations.coco.json',
    metric=['bbox', 'segm']
)

# =========================
# Training / Validation / Testing Loops
# =========================
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# =========================
# Hooks
# =========================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP'
    ),
    logger=dict(type='LoggerHook', interval=50)
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/segm_mAP',
        rule='greater',
        min_delta=0.002,
        patience=6,
        strict=False,
        check_finite=True
    )
]

# =========================
# Optimizer + Regularization
# =========================
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.002,
        momentum=0.9,
        weight_decay=0.0005
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# =========================
# Learning Rate Scheduler
# =========================
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[30, 40],
        gamma=0.1
    )
]

# =========================
# Runtime
# =========================
randomness = dict(seed=42, deterministic=False)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

work_dir = 'C:/Users/gelob/Desktop/mmdetection/work_dirs/solov2_eggplant'