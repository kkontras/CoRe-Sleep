import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from graphs.models.unet_parts import Down

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Stft_model_1(nn.Module):

    def __init__(self,inplanes=9) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, 64, stride=1)
        self.block1 = BasicBlock(64,64)
        self.down1 = Down(64,128)
        self.block2 = BasicBlock(128,128)
        self.down2 = Down(128,256)
        self.block3 = BasicBlock(256,256)
        self.conv4 = conv3x3(256, 512, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(512*9*5,1024)
        self.fc2 = nn.Linear(1024,128)
        self.fc3 = nn.Linear(128,2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x) -> Tensor:
        x = self.conv1(x.float())
        x = self.block1(x)
        x = self.down1(x)
        x = self.dropout(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.dropout(x)
        x = self.block3(x)
        x = self.conv4(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(torch.flatten(x, start_dim=1)))
        x = self.relu(self.fc2(self.dropout(x)))
        x = self.logsoftmax(self.fc3(x))
        return x
