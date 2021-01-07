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
import mxnet as mx
import numpy as np
import os
from common import data, fit, modelzoo
from datetime import datetime as dt

nowtime = dt.now().strftime('%Y-%m-%d-%H:%M:%S')

logging.basicConfig(
    filename=os.path.join(
        os.getcwd(), 'logs/20201109_geligao-densenet201-8911-finetune-' +
        str(nowtime) + '-log.txt'),
    level=logging.DEBUG)


def get_fine_tune_model(symbol,
                        arg_params,
                        num_classes,
                        layer_name,
                        dtype='float32'):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    print(all_layers)
    net = all_layers[layer_name + '_output']

    net = mx.symbol.FullyConnected(
        data=net, num_hidden=2048, name='fc1', attr={'lr_mult': '10.0'})  #
    prob = 0.5
    do1 = mx.sym.Dropout(data=net, name='do1', p=prob)
    logging.info('dropout probability:' + str(prob))
    net = mx.sym.Activation(data=do1, act_type='relu', name='relu_fc1')
    net = mx.symbol.FullyConnected(
        data=net, num_hidden=1024, name='fc2', attr={'lr_mult': '10.0'})  #
    prob = 0.5
    do2 = mx.sym.Dropout(data=net, name='do2', p=prob)
    net = mx.sym.Activation(data=do2, act_type='relu', name='relu_fc2')

    net = mx.symbol.FullyConnected(
        data=net, num_hidden=num_classes, name='fc3', attr={'lr_mult':
                                                            '10.0'})  #

    logging.info('dropout probability:' + str(prob))
    if dtype == 'float16':
        net = mx.sym.Cast(data=net, dtype=np.float32)

    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc' not in k})
    return (net, new_args)


if __name__ == '__main__':
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)

    logging.getLogger('').addHandler(console)
    # parse args
    parser = argparse.ArgumentParser(
        description='fine-tune a dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train = fit.add_fit_args(parser)
    data.add_data_args(parser)
    aug = data.add_data_aug_args(parser)
    parser.add_argument(
        '--pretrained-model',
        default='',
        type=str,
        help='the pre-trained model. can be prefix of local model files prefix \
                        or a model name from common/modelzoo')
    parser.add_argument(
        '--layer-before-fullc',
        type=str,
        default='pool5',
        help='the name of the layer before the last fullc layer')

    # use less augmentations for fine-tune.
    # by default here it uses no augmentations

    # use a small learning rate and less regularizations
    parser.set_defaults(
        # network
        # network        = 'resnet',
        # num_layers     = 110,
        # data
        data_train='',
        data_val='',
        num_classes=5532,
        num_examples=2873014,
        image_shape='3,224,224',
        pad_size=4,
        # train
        batch_size=20,
        num_epochs=16,
        lr=.001,
        lr_factor=0.1,
        lr_step_epochs='8,11,14',
    )
    args = parser.parse_args()

    # load pretrained model and params
    dir_path = os.path.dirname(os.path.realpath(__file__))
    (prefix, epoch) = modelzoo.download_model(args.pretrained_model,
                                              os.path.join(dir_path, 'model'))
    print('dir_path:', dir_path)
    if prefix is None:
        (prefix, epoch) = (args.pretrained_model, 0)
    # print('prefix:',prefix)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    if args.dtype != 'float32':
        # load symbol of trained network,
        # so we can cast it to support other dtype
        # fine tuning a network in a datatype
        # which was not used for training originally,
        # requires access to the code used to
        # generate the symbol used to train that model.
        # we then need to modify the symbol to add a layer at the beginning
        # to cast data to that dtype.
        # We also need to cast output of layers before softmax
        # to float32 so that softmax can still be in float32.
        # if the network chosen from symols/
        # folder doesn't have cast for the new datatype,
        # it will still train in fp32
        if args.network not in [
                'inception-v3', 'inception-v4', 'resnet-v1', 'resnet',
                'resnext', 'vgg'
        ]:
            raise ValueError('Given network does not have support \
                for dtypes other than float32. \
                Please add a cast layer at the \
                beginning to train in that mode.')
        from importlib import import_module
        net = import_module('symbols.' + args.network)
        sym = net.get_symbol(**vars(args))

    # remove the last fullc layer and add a new softmax layer
    print('################: ', args.layer_before_fullc)
    (new_sym,
     new_args) = get_fine_tune_model(sym, arg_params, args.num_classes,
                                     args.layer_before_fullc, args.dtype)

    # train
    fit.fit(
        args=args,
        network=new_sym,
        data_loader=data.get_rec_iter,
        arg_params=new_args,
        aux_params=aux_params)
