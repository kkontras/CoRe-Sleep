"""
Cross Entropy 2D for CondenseNet
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class MSELoss(nn.Module):
    def __init__(self, config=None):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)