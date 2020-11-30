import logging
import sys

import mxnet as mx
from mxnet import npx, autograd
import gluoncv
import mxnet.gluon as gluon
from mxnet.gluon import data, loss as gloss

from src.data.data_handler import DatasetRFS
from src.models.resnet import FactoryModels


class TrainParameters:
    """
    Contains the hyper parameters for the training process

    Parameters
    ----------
    :param batch_size: ~ (int)
    :param ctx: Computational context (npx.cpu(0), npx.gpu(0), etc)
    :param num_epoch: Number of total training epochs (int)
    :param weight_decay: ~ (double)
    :param momentum_train: ~ (double, < 1.0)
    :param num_workers: CPU threads used to extract data (int)
    :param ident: Identification of the present training process (String)
    :param train_baselr: Basic learning rate (double)
    :param train_poly_power: Polynomial power for the adaption of the learning rate over epochs (int)
    :param criterion: Loss criterion (mxnet.gluon.loss.Loss)
    :param kv: Handling multiple GPU instances (mx.kv)
    :param optimizing: Name of the optimizing process (String)
    """
    def __init__(self, batch_size, ctx, num_epoch, weight_decay, momentum_train, num_workers, ident,
                 train_baselr, train_poly_power, criterion, kv, optimizing):
        self.batch_size = batch_size
        self.ctx = ctx
        self.num_epoch = num_epoch
        self.weight_decay = weight_decay
        self.momentum_train = momentum_train
        self.num_workers = num_workers
        self.ident = ident
        self.train_baselr = train_baselr
        self.train_poly_power = train_poly_power
        self.criterion = criterion
        self.kv = kv
        self.optimizing = optimizing


class FactoryParameters:

    @staticmethod
    def get_best_practice(ident, ctx=npx.cpu(0)):
        """
        Contains the hyper parameters for the training process

        Parameters
        ----------
        :param ident: Name of the training run
        :param ctx: Computational context
        """
        criterion = gloss.SoftmaxCrossEntropyLoss(sparse_label=False)
        kv = mx.kv.create('device')
        para = TrainParameters(batch_size=10, ctx=ctx, num_epoch=500, weight_decay=0.01, momentum_train=0.9,
                               num_workers=1, ident=ident, train_baselr=0.0001, train_poly_power=0.9,
                               criterion=criterion, kv=kv, optimizing='sgd')
        return para


class Training:
    """
    Class that handles the training process as well as hyper parameters

    Parameters
    ----------
    :param param: TrainParameters class
    :param data_root: Path to the data file
    """
    def __init__(self, param, data_root):
        self.param = param
        self.data_root = data_root
        self.log = None

    def start_training(self, net):
        """
        Execute the training

        Parameters
        ----------
        :param net: Model to be trained
        """
        train_dataset = DatasetRFS(self.data_root)
        train_data = data.DataLoader(train_dataset, batch_size=self.param.batch_size, shuffle=True,
                                     last_batch='rollover',
                                     num_workers=self.param.num_workers)
        net.collect_params().reset_ctx(ctx=self.param.ctx)

        lr_scheduler = gluoncv.utils.LRScheduler('poly', base_lr=self.param.train_baselr, nepochs=self.param.num_epoch,
                                      iters_per_epoch=len(train_data), power=self.param.train_poly_power)

        optimizer = gluon.Trainer(net.collect_params(), self.param.optimizing,
                                  {'lr_scheduler': lr_scheduler,
                                   'wd': self.param.weight_decay,
                                   'momentum': self.param.momentum_train,
                                   'multi_precision': True},
                                  kvstore=self.param.kv)
        self.set_logging()
        Training.train(net=net, train_data=train_data, optimizer=optimizer, param=self.param, log=self.log)

    def set_logging(self, **kwargs):
        """Define in this function how the output of the training process will be handled"""
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        self.log = logging.getLogger('train_output')

    @staticmethod
    def train(net, train_data, optimizer, param, log):
        """Define in this function how the output of the training process will be handled"""
        for epoch in range(param.num_epoch):
            train_loss = 0.0
            for i, (input_image, target) in enumerate(train_data):
                data_ctx = input_image.copyto(param.ctx)
                target_ctx = target.copyto(param.ctx)
                with autograd.record(True):
                    outputs = net(data_ctx)
                    losses = param.criterion(outputs, target_ctx)
                    mx.nd.waitall()
                    autograd.backward(losses)
                optimizer.step(param.batch_size)
                for loss in losses:
                    train_loss += loss.asnumpy()[0] / len(losses)
                if log is not None:
                    log.info('Epoch %d, batch %d, training loss %.3f' % (epoch, i, train_loss / (i + 1)))
            Training.save_model(net=net, param=param)

    @staticmethod
    def save_model(net, param, **kwargs):
        """Define in this function how the model will be saved after each epoch"""
        net.save_parameters('net_' + param.ident + '.params')

    @staticmethod
    def start_standard_run():
        """Define here what the standard, baseline procedure is"""
        import os
        import pathlib
        train_object = Training(param=FactoryParameters.get_best_practice(ident='test_run'), data_root=
                                os.path.join(pathlib.Path(__file__).parent.parent.parent.absolute(), 'sampledata'))
        train_object.start_training(net=FactoryModels.get_standard_model())
