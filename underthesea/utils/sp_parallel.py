# -*- coding: utf-8 -*-

import os
from random import Random

import torch
import torch.distributed as dist
import torch.nn as nn


class DistributedDataParallel(nn.parallel.DistributedDataParallel):

    def __init__(self, module, **kwargs):
        super().__init__(module, **kwargs)

    def __getattr__(self, name):
        wrapped = super().__getattr__('module')
        if hasattr(wrapped, name):
            return getattr(wrapped, name)
        return super().__getattr__(name)


def init_device(device, backend='nccl', host=None, port=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    if torch.cuda.device_count() > 1:
        host = host or os.environ.get('MASTER_ADDR', 'localhost')
        port = port or os.environ.get('MASTER_PORT', str(Random(0).randint(10000, 20000)))
        os.environ['MASTER_ADDR'] = host
        os.environ['MASTER_PORT'] = port
        dist.init_process_group(backend)
        torch.cuda.set_device(dist.get_rank())


def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0
