_base_ = [
    '../_base_/models/resnest101.py',
    '../_base_/datasets/imagenet_bs32_lmdb_balance.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
