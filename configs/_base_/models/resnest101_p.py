# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNeSt',
        depth=101,
        num_stages=4,
        stem_channels=128,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5532,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
