from typing import List

import torch
from pytorch_toolbelt.modules import (
    DecoderModule,
    PPMDecoder,
    conv1x1,
    ResidualDeconvolutionUpsample2d,
    ACT_RELU,
    instantiate_activation_block,
    ACT_SILU,
)
from torch import nn
from torch.nn import functional as F

__all__ = ["PPLinkNetDecoder"]


class PPMBlock(nn.Module):
    def __init__(self, channels: int, reduction_dim: int, bins=(3, 5, 7, 9), activation=ACT_RELU):
        super().__init__()
        self.output_channels = channels + reduction_dim * 4
        ppm = []
        for kernel_size in bins:
            ppm.append(
                nn.Sequential(
                    nn.AvgPool2d(kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2),
                    nn.Conv2d(channels, reduction_dim, kernel_size=(1, 1), bias=False),
                    nn.BatchNorm2d(reduction_dim),
                    instantiate_activation_block(activation, inplace=True),
                )
            )
        self.ppm = nn.ModuleList(ppm)

    def forward(self, x):
        input_size = x.size()[2:]
        ppm_out = [x]
        for pool_scale in self.ppm:
            input_pooled = pool_scale(x)
            input_pooled = F.interpolate(input_pooled, size=input_size, mode="bilinear", align_corners=True)
            ppm_out.append(input_pooled)
        ppm_out = torch.cat(ppm_out, dim=1)
        return ppm_out


class PPLinkDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=ACT_RELU):
        super().__init__()
        self.up = ResidualDeconvolutionUpsample2d(in_channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            instantiate_activation_block(activation, inplace=True),
        )

        self.conv2 = nn.Sequential(
            conv1x1(in_channels // 4, out_channels), nn.BatchNorm2d(out_channels), instantiate_activation_block(activation, inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)
        return x


class PPLinkNetDecoder(DecoderModule):
    def __init__(self, feature_maps: List[int], activation=ACT_RELU, bins=(3, 5, 7, 9)):
        super().__init__()

        self.center = PPMBlock(feature_maps[-1], reduction_dim=int(feature_maps[-1] / len(bins)), bins=bins, activation=activation)
        decoders = []
        in_ch = self.center.output_channels
        for ch in reversed(feature_maps[:-1]):
            decoders.append(PPLinkDecoderBlock(in_ch, ch, activation=activation))
            in_ch = ch
        self.decoders = nn.ModuleList(decoders)
        self.channels = [feature_maps[0]]
        self.bn = nn.BatchNorm2d(feature_maps[0])

    def forward(self, features):
        feature_maps, last_fm = features[:-1], features[-1]
        x = self.center(last_fm)
        for enc_fm, decoder_block in zip(reversed(feature_maps), self.decoders):
            x = decoder_block(x) + enc_fm

        return [self.bn(x)]


def main():
    channels = [64, 128, 256, 512, 2048]
    net = PPLinkNetDecoder(channels, activation=ACT_SILU).eval()

    inputs = [
        torch.randn((4, 64, 256, 256)),
        torch.randn((4, 128, 128, 128)),
        torch.randn((4, 256, 64, 64)),
        torch.randn((4, 512, 32, 32)),
        torch.randn((4, 2048, 16, 16)),
    ]

    outputs = net(inputs)
    print(outputs[0].size(), outputs[0].mean(), outputs[0].std())


if __name__ == "__main__":
    main()
