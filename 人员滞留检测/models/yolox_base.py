#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.distributed as dist
import torch.nn as nn

import os
import random
from .other_common import YOLOPAFPN, YOLOX, YOLOXHead
from .base_exp import BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 1
        self.depth = 0.67
        self.width = 0.75

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.input_size = (800, 1440)
        self.random_size = (18, 32)

        # --------------- transform config ----------------- #
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mscale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True

        # -----------------  testing config ------------------ #
        self.test_size = (800, 1440)
        self.test_conf = 0.25
        self.nmsthre = 0.6
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

    def get_model(self):
        #from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
    
    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = data_loader.change_input_dim(
            multiple=(tensor[0].item(), tensor[1].item()), random_range=None
        )
        return input_size

