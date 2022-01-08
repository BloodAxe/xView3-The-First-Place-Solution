from functools import partial

import torch
from pytorch_toolbelt.modules import DepthwiseSeparableConv2d, ABN
from timm.models.efficientnet_blocks import InvertedResidual
from timm.models.layers import EffectiveSEModule
from torch import nn
from pytorch_toolbelt.modules import (
    ACT_RELU,
    decoders as D,
    UnetBlock,
    get_activation_block,
    ABN,
    EncoderModule,
)

__all__ = ["ResidualUnetBlock", "InvertedResidual", "DenseNetUnetBlock", "AdditionalEncoderStage"]


def get_unet_block(block_name: str, activation=ACT_RELU):
    if block_name == "ResidualUnetBlock":
        return partial(ResidualUnetBlock, abn_block=partial(ABN, activation=activation))
    elif block_name == "IRBlock":
        return partial(IRBlock, act_block=get_activation_block(activation))
    elif block_name == "DenseNetUnetBlock":
        return partial(DenseNetUnetBlock, abn_block=partial(ABN, activation=activation))
    elif block_name == "UnetBlock":
        return partial(UnetBlock, abn_block=partial(ABN, activation=activation))


class ResidualUnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, abn_block=ABN):
        super().__init__()
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv1 = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.abn1 = abn_block(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.abn2 = abn_block(out_channels)
        self.conv3 = DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.abn3 = abn_block(out_channels)

    def forward(self, x):
        residual = self.identity(x)

        x = self.conv1(x)
        x = self.abn1(x)

        x = self.conv2(x)
        x = self.abn2(x)

        x = self.conv3(x)
        x = self.abn3(x)

        return x + residual


class DenseNetUnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, abn_block=ABN):
        super().__init__()
        self.conv1 = ResidualUnetBlock(in_channels, out_channels, abn_block=abn_block)
        self.conv2 = ResidualUnetBlock(in_channels + out_channels, out_channels, abn_block=abn_block)

    def forward(self, x):
        y = self.conv1(x)
        x = self.conv2(torch.cat([x, y], dim=1))
        return x


class AdditionalEncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_layer=nn.ReLU):
        super().__init__()
        self.ir_block1 = InvertedResidual(in_channels, out_channels, act_layer=act_layer, stride=2)
        self.ir_block2 = InvertedResidual(out_channels, out_channels, act_layer=act_layer, dilation=2, se_layer=EffectiveSEModule)

    def forward(self, x):
        x = self.ir_block1(x)
        return self.ir_block2(x)


class IRBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_block=nn.ReLU, stride=1):
        super().__init__()
        self.ir_block1 = InvertedResidual(in_channels, out_channels, act_layer=act_block, stride=stride)
        self.ir_block2 = InvertedResidual(out_channels, out_channels, act_layer=act_block, se_layer=EffectiveSEModule)

    def forward(self, x):
        x = self.ir_block1(x)
        x = self.ir_block2(x)
        return x
