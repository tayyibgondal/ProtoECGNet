import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fastai.layers import Flatten
from basic_conv1d import create_head1d

###############################################################################################
# Standard resnet

def conv(in_planes, out_planes, stride=1, kernel_size=3):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)


class BasicBlock1d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, kernel_size=[3,3], downsample=None):
        super().__init__()

        if isinstance(kernel_size, int): 
            kernel_size = [kernel_size, kernel_size // 2 + 1]

        self.conv1 = conv(inplanes, planes, stride=stride, kernel_size=kernel_size[0])
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes, kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck1d(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, kernel_size=3, downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size-1)//2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)  # Bottleneck
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # Intermediary
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # Expand volume/channels
        out = self.bn3(out)

        if downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet1d(nn.Module):
    '''1d adaptation of the torchvision resnet'''
    def __init__(self, block, layers, kernel_size=3, num_classes=2, input_channels=3, inplanes=64, fix_feature_dim=True, kernel_size_stem=None, stride_stem=2, pooling_stem=True, stride=2, lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
        super().__init__()
        self.inplanes = inplanes

        # Define layers
        layers_tmp = []

        if kernel_size_stem is None:
            kernel_size_stem = kernel_size[0] if isinstance(kernel_size, list) else kernel_size
        
        # Stem
        layers_tmp.append(nn.Conv1d(input_channels, inplanes, kernel_size=kernel_size_stem, stride=stride_stem, padding=(kernel_size_stem-1)//2, bias=False))
        layers_tmp.append(nn.BatchNorm1d(inplanes))
        layers_tmp.append(nn.ReLU(inplace=True))
        if pooling_stem:
            layers_tmp.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        
        # Backbone
        for i, l in enumerate(layers):
            if i == 0:
                layers_tmp.append(self._make_layer(block, inplanes, layers[0], kernel_size=kernel_size))
            else:
                layers_tmp.append(self._make_layer(block, inplanes if fix_feature_dim else (2**i) * inplanes, layers[i], stride=stride, kernel_size=kernel_size))
        
        self.backbone = nn.Sequential(*layers_tmp)
        
    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=3):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, kernel_size, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def get_layer_groups(self):
        return (self.backbone, )

    def get_output_layer(self):
        return None  # No classification head
    
    def set_output_layer(self, x):
        pass  # No classification head

    def conv_info(self):
        """
        Returns three lists: kernel_sizes, strides, paddings for each Conv1d layer in the model.
        """
        kernel_sizes = []
        strides = []
        paddings = []
        
        for layer in self.backbone:
            if isinstance(layer, nn.Sequential):
                for block in layer:
                    if isinstance(block, (BasicBlock1d, Bottleneck1d)):
                        kernel_sizes.append(block.conv1.kernel_size[0])
                        strides.append(block.conv1.stride[0])
                        paddings.append(block.conv1.padding[0])
                        
                        kernel_sizes.append(block.conv2.kernel_size[0])
                        strides.append(block.conv2.stride[0])
                        paddings.append(block.conv2.padding[0])
                        
                        if isinstance(block, Bottleneck1d):
                            kernel_sizes.append(block.conv3.kernel_size[0])
                            strides.append(block.conv3.stride[0])
                            paddings.append(block.conv3.padding[0])
        
        return kernel_sizes, strides, paddings


def resnet1d_wang(**kwargs):
    if not "kernel_size" in kwargs.keys():
        kwargs["kernel_size"] = [5, 3]
    if not "kernel_size_stem" in kwargs.keys():
        kwargs["kernel_size_stem"] = 7
    if not "stride_stem" in kwargs.keys():
        kwargs["stride_stem"] = 1
    if not "pooling_stem" in kwargs.keys():
        kwargs["pooling_stem"] = False
    if not "inplanes" in kwargs.keys():
        kwargs["inplanes"] = 128

    return ResNet1d(BasicBlock1d, [1, 1, 1], **kwargs)
