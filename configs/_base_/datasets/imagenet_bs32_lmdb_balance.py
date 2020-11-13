# dataset settings
dataset_type = 'LMDBDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={
            'backend': 'lmdb',
            'db_path': 'data/lmdb/train.lmdb'
        }),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={
            'backend': 'lmdb',
            'db_path': 'data/lmdb/train.lmdb'
        }),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            data_prefix='',
            ann_file='data/lmdb/train.lmdb',
            pipeline=train_pipeline)),
    val=dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            data_prefix='',
            ann_file='data/lmdb/val.lmdb',
            pipeline=train_pipeline)),
    test=dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            data_prefix='',
            ann_file='data/lmdb/val.lmdb',
            pipeline=train_pipeline)))
evaluation = dict(interval=1, metric='accuracy')
