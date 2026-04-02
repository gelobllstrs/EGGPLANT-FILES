_base_ = 'rtmdet-ins_tiny_8xb32-300e_coco.py'
# If this base file is not found in your version, tell me the exact files inside configs/rtmdet

# =========================
# Dataset
# =========================
dataset_type = 'CocoDataset'
data_root = 'C:/Users/gelob/Desktop/mmdetection format dataset/'

classes = ('infested', 'non-infested')
metainfo = dict(classes=classes)

# =========================
# Model
# RTMDet-Ins uses only bbox_head for num_classes
# Do NOT add mask_head here
# =========================
model = dict(
    bbox_head=dict(
        num_classes=2
    )
)

# =========================
# Pipelines
# Keep light because Roboflow already augmented your dataset
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
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
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
        metainfo=metainfo,
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
        metainfo=metainfo,
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
# Training / Validation / Testing
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
        type='AdamW',
        lr=0.0002,
        weight_decay=0.0005
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# =========================
# LR Scheduler
# =========================
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=50,
        eta_min=1e-6,
        by_epoch=True,
        begin=0,
        end=50
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

work_dir = 'C:/Users/gelob/Desktop/mmdetection/work_dirs/rtmdet_ins_eggplant'