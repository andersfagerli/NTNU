import torch
from torch import nn, Tensor
from torchvision import models
from typing import Type, Any, Callable, Union, List, Optional

# ----------------------------------------------------------------------------------------------------------------------#
# Code taken from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# in order to build extra layers with ResNet architecture

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


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

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
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# ----------------------------------------------------------------------------------------------------------------------#

class ResNetRDD(nn.Module):
    """
    This is a ResNet backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg, block: Type[Union[BasicBlock, Bottleneck]] = BasicBlock):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        # ResNet parameters
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        
        self.model = models.resnet18(pretrained=cfg.MODEL.BACKBONE.PRETRAINED)

        if cfg.MODEL.BACKBONE.FREEZE:
            assert cfg.MODEL.BACKBONE.PRETRAINED == True,\
            f"Shouldn't freeze untrained network"

            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze layers part of feature map
            for param in self.model.layer3.parameters(): 
                param.requires_grad = True
            for param in self.model.layer4.parameters():
                param.requires_grad = True

        # Extra feature maps outside resnet backbone
        self.extra1 = self._make_layer(block, output_channels[1], output_channels[2], 2, 2)
        self.extra2 = self._make_layer(block, output_channels[2], output_channels[3], 2, 2)
        self.extra3 = self._make_layer(block, output_channels[3], output_channels[4], 2, 2)
        self.extra4 = self._make_layer(block, output_channels[4], output_channels[5], 2, 3)

        self.feature_maps = [
            self.model.layer3,
            self.model.layer4,
            self.extra1,
            self.extra2,
            self.extra3,
            self.extra4
        ]

        # Weight initialization for extra layers
        for feature_map in self.feature_maps[2:]:
            for module in feature_map.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)
                elif isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)

        params = 0
        params += sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        for feature_map in self.feature_maps[2:]:
            params += sum(p.numel() for p in feature_map.parameters() if p.requires_grad)
        print("Total trainable params: ", params)

    # Taken from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    # Modified: Defines input and output channels explicitely, instead of expanding each time
    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], in_channels: int, out_channels: int,
                    blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample, self.groups,
                            self.base_width, previous_dilation, nn.BatchNorm2d))
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=nn.BatchNorm2d))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        # Propagate input through first layers
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)

        # Propagate through feature maps and save each output
        out_features = []
        out_features.append(self.feature_maps[0](x))
        for i in range(1, len(self.feature_maps)):
            out_features.append(self.feature_maps[i](out_features[i-1]))

        # Check dimensions are correct
        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        
        return tuple(out_features)

       