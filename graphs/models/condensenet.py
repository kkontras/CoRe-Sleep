"""
CondenseNet Model
name: condensenet.py
date: May 2018
"""
import torch
import torch.nn as nn
import json
from easydict import EasyDict as edict
import numpy as np

from graphs.weights_initializer import init_model_weights
from graphs.models.custom_layers.denseblock import DenseBlock
from graphs.models.custom_layers.learnedgroupconv import LearnedGroupConv

class CondenseNet(nn.Module):
    def __init__(obj, config):
        super().__init__()
        obj.config = config

        obj.stages = obj.config.stages
        obj.growth_rate = obj.config.growth_rate
        assert len(obj.stages) == len(obj.growth_rate)

        obj.init_stride = obj.config.init_stride
        obj.pool_size = obj.config.pool_size
        obj.num_classes = obj.config.num_classes

        obj.progress = 0.0
        obj.num_filters = 2 * obj.growth_rate[0]
        """
        Initializing layers
        """
        obj.transition_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        obj.pool = nn.AvgPool2d(obj.pool_size)
        obj.relu = nn.ReLU(inplace=True)

        obj.init_conv = nn.Conv2d(in_channels=obj.config.input_channels, out_channels=obj.num_filters, kernel_size=3, stride=obj.init_stride, padding=1, bias=False)

        obj.denseblock_one = DenseBlock(num_layers=obj.stages[0], in_channels= obj.num_filters, growth_rate=obj.growth_rate[0], config=obj.config)

        obj.num_filters += obj.stages[0] * obj.growth_rate[0]

        obj.denseblock_two = DenseBlock(num_layers=obj.stages[1], in_channels= obj.num_filters, growth_rate=obj.growth_rate[1], config=obj.config)

        obj.num_filters += obj.stages[1] * obj.growth_rate[1]

        obj.denseblock_three = DenseBlock(num_layers=obj.stages[2], in_channels= obj.num_filters, growth_rate=obj.growth_rate[2], config=obj.config)

        obj.num_filters += obj.stages[2] * obj.growth_rate[2]
        self.batch_norm = nn.BatchNorm2d(self.num_filters)

        self.classifier = nn.Linear(self.num_filters, self.num_classes)

        self.apply(init_model_weights)

    def forward(self, x, progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress

        x = self.init_conv(x)

        x = self.denseblock_one(x)
        x = self.transition_pool(x)

        x = self.denseblock_two(x)
        x = self.transition_pool(x)

        x = self.denseblock_three(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        out = self.classifier(x)

        return out

"""
#########################
Model Architecture:
#########################

Input: (N, 32, 32, 3)

- Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
DenseBlock(num_layers=14, in_channels=16, growth_rate=8)
- AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True)
DenseBlock(num_layers=14, in_channels=128, growth_rate=16)
- AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True)
DenseBlock(num_layers=14, in_channels=352, growth_rate=32)
- BatchNorm2d(800, eps=1e-05, momentum=0.1, affine=True)
- ReLU(inplace)
- AvgPool2d(kernel_size=8, stride=8, padding=0, ceil_mode=False, count_include_pad=True)
- Linear(in_features=800, out_features=10, bias=True)
"""
