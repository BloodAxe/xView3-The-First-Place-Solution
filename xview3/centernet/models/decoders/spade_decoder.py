from typing import List, Union, Type

import torch
from torch import nn, Tensor
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from pytorch_toolbelt.utils import count_parameters
from pytorch_toolbelt.modules import (
    EncoderModule,
    DecoderModule,
    get_activation_block,
    ACT_RELU,
    instantiate_activation_block,
)


__all__ = ["SPADEDecoder"]


def conv3x3(a, b, bias=True):
    return nn.Conv2d(a, b, kernel_size=(3, 3), padding=1, bias=bias)


class SPADELayer(nn.Module):
    def __init__(
        self,
        input_channels: int,
        mod_input_channels: int,
    ):
        super().__init__()

        self.conv = nn.Sequential(conv3x3(mod_input_channels, input_channels, bias=False), nn.BatchNorm2d(input_channels))
        self.conv_gamma = nn.Sequential(conv3x3(input_channels, input_channels, bias=False), nn.BatchNorm2d(input_channels))
        self.conv_beta = nn.Sequential(conv3x3(input_channels, input_channels, bias=False), nn.BatchNorm2d(input_channels))

    def _init_weights(self):
        nn.init.orthogonal_(self.conv.weight, 0.1)
        nn.init.orthogonal_(self.conv_gamma.weight, 0.1)
        nn.init.orthogonal_(self.conv_beta.weight, 0.1)

    def forward(self, x, mod_input):
        mod_input = self.conv(mod_input)
        gamma = self.conv_gamma(mod_input)
        beta = self.conv_beta(mod_input)

        x = x * (1 + gamma) + beta
        if torch.isnan(x).any():
            print("Here")

        return x


class SPADEBlock(nn.Module):
    def __init__(self, input_channels: int, mod_input_channels: int, activation: str = ACT_RELU):
        super().__init__()
        self.spade_1 = SPADELayer(input_channels, mod_input_channels)
        self.conv_1 = nn.Sequential(
            conv3x3(input_channels, input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            instantiate_activation_block(activation),
        )
        self.spade_2 = SPADELayer(input_channels, mod_input_channels)
        self.conv_2 = nn.Sequential(
            conv3x3(input_channels, input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            instantiate_activation_block(activation),
        )
        self.act = instantiate_activation_block(activation)

    def forward(self, x, mod_input):
        x = self.act(self.spade_1(x, mod_input))
        if torch.isnan(x).any():
            print("Here")
        x = self.conv_1(x)
        if torch.isnan(x).any():
            print("Here")
        x = self.act(self.spade_2(x, mod_input))
        if torch.isnan(x).any():
            print("Here")
        x = self.conv_2(x)
        if torch.isnan(x).any():
            print("Here")

        return x


class SPADEDecoderBlock(nn.Module):
    def __init__(self, encoder_channels: int, decoder_channels: int, output_channels: int, activation=ACT_RELU):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.output_channels = output_channels
        self.upsample = nn.Sequential(nn.Conv2d(decoder_channels, output_channels * 4, kernel_size=(3, 3), padding=1), nn.PixelShuffle(2))

        self.spade = SPADEBlock(output_channels, encoder_channels, activation)

    def __repr__(self):
        return f"SPADEDecoderBlock(encoder_channels={self.encoder_channels},decoder_channels={self.decoder_channels},output_channels={self.output_channels})"

    def forward(self, encoder: Tensor, decoder: Tensor) -> Tensor:
        decoder = self.upsample(decoder)
        y = self.spade(decoder, encoder)
        if torch.isnan(y).any():
            print("Here")

        return y


class SPADEDecoder(DecoderModule):
    def __init__(self, feature_maps: List[int], decoder_features: List[int], activation=ACT_RELU):
        super().__init__()

        blocks = []
        num_blocks = len(feature_maps) - 1  # Number of outputs is one less than encoder layers
        decoder_channels = feature_maps[-1]

        for block_index in reversed(range(num_blocks)):
            features_from_encoder = feature_maps[block_index]
            out_channels = decoder_features[block_index]
            blocks.append(SPADEDecoderBlock(features_from_encoder, decoder_channels, out_channels, activation=activation))
            decoder_channels = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.output_filters = decoder_features

    @property
    @torch.jit.unused
    def channels(self) -> List[int]:
        return self.output_filters

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        x = feature_maps[-1]
        outputs = []
        num_feature_maps = len(feature_maps)
        for index, decoder_block in enumerate(self.blocks):
            encoder_input = feature_maps[num_feature_maps - index - 2]

            x = decoder_block(encoder_input, x)
            outputs.append(x)

        # Returns last of tensors in same order as input (fine-to-coarse)
        return outputs[-1]


if __name__ == "__main__":
    torch.set_anomaly_enabled(True)

    with torch.no_grad():
        x = [
            torch.randn((2, 32, 256, 256)),
            torch.randn((2, 64, 128, 128)),
            torch.randn((2, 128, 64, 64)),
            torch.randn((2, 256, 32, 32)),
            torch.randn((2, 512, 16, 16)),
        ]

        decoder = SPADEDecoder([32, 64, 128, 256, 512], [48, 96, 192, 384]).eval()

        print(count_parameters(decoder))
        with torch.cuda.amp.autocast(True):
            y = decoder(x)

        for yi in y:
            print(yi.size(), yi.mean(), yi.std())
