_base_ = [
    '../_base_/models/resnest101_p.py',
    '../_base_/datasets/imagenet_bs32_lmdb_balance_batch40.py',
    '../_base_/schedules/imagenet_bs256_10e.py', '../_base_/default_runtime.py'
]
