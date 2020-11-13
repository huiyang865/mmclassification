_base_ = [
    '../_base_/models/resnest101.py', './imagenet_bs32_lmdb_balance.py',
    './imagenet_bs256.py', './default_runtime.py'
]
