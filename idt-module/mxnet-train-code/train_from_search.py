# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import logging
import os
from common import data, fit
from common.util import download_file
from datetime import datetime as dt

nowtime = dt.now().strftime('%Y-%m-%d-%H:%M:%S')

logging.basicConfig(
    filename=os.path.join(os.getcwd(), 'train_from_search_log/' +
                          str(nowtime) + '-log.txt'),
    level=logging.DEBUG)


def download_cifar10():
    data_dir = 'data'
    fnames = (os.path.join(data_dir, 'cifar10_train.rec'),
              os.path.join(data_dir, 'cifar10_val.rec'))
    download_file('http://data.mxnet.io/data/cifar10/cifar10_val.rec',
                  fnames[1])
    download_file('http://data.mxnet.io/data/cifar10/cifar10_train.rec',
                  fnames[0])
    return fnames


def set_cifar_aug(aug):
    aug.set_defaults(rgb_mean='125.307,122.961,113.8575', rgb_std='1,1,1')
    aug.set_defaults(random_mirror=1, pad=4, fill_value=0, random_crop=1)
    aug.set_defaults(min_random_size=32, max_random_size=32)


if __name__ == '__main__':
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)

    logging.getLogger('').addHandler(console)
    # download data
    (train_fname, val_fname) = download_cifar10()

    # parse args
    parser = argparse.ArgumentParser(
        description='train_from_search',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    # uncomment to set standard cifar augmentations
    # set_cifar_aug(parser)
    parser.set_defaults(
        # network
        network='resnet',
        num_layers=50,
        # data
        data_train='data/train.rec',
        data_val='data/val.rec',
        num_classes=187,
        num_examples=104588,
        image_shape='3,224,224',
        pad_size=4,
        # train
        batch_size=112,
        num_epochs=500,
        lr=.005,
        lr_step_epochs='80,100,200',
    )
    args = parser.parse_args()
    logging.info('network:' + args.network + str(args.num_layers))
    # load network
    from importlib import import_module
    net = import_module('symbols.' + args.network)
    sym = net.get_symbol(**vars(args))
    print((net))
    # train
    fit.fit(args, sym, data.get_rec_iter)
