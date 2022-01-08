from typing import List

import torch
from omegaconf import DictConfig
from pytorch_toolbelt.modules import (
    ACT_RELU,
    decoders as D,
    EncoderModule,
    ResidualDeconvolutionUpsample2d,
)
from torch import nn

from .unet_blocks import get_unet_block

__all__ = ["SiameseUNetModel"]


class SiameseUNetModel(nn.Module):
    def __init__(
        self,
        encoders: List[EncoderModule],
        decoder_features: List[int],
        head: nn.Module,
        box_coder,
        input_upsample_factor: int = 1,
        activation=ACT_RELU,
        block_type="UnetBlock",
        upsample_type="bilinear",
    ):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.output_stride = encoders[0].strides[0]
        self.initial_upsample = (
            nn.UpsamplingBilinear2d(scale_factor=input_upsample_factor) if input_upsample_factor != 1 else nn.Identity()
        )

        unet_block = get_unet_block(block_type, activation=activation)

        upsample_block = {
            "nearest": nn.UpsamplingNearest2d,
            "bilinear": nn.UpsamplingBilinear2d,
            "rdtsc": ResidualDeconvolutionUpsample2d,
            "shuffle": nn.PixelShuffle,
        }[upsample_type]

        encoder_channels = [ch * len(encoders) for ch in encoders[0].channels]
        self.decoder = D.UNetDecoder(
            encoder_channels,
            decoder_features,
            unet_block=unet_block,
            upsample_block=upsample_block,
        )
        self.head = head
        self.box_coder = box_coder

    def concat_features(self, all_feature_maps):
        num_channels = len(all_feature_maps)
        num_layers = len(all_feature_maps[0])

        concatenated_features = []
        for layer in range(num_layers):
            fm = torch.cat([all_feature_maps[channel][layer] for channel in range(num_channels)], dim=1)
            concatenated_features.append(fm)
        return concatenated_features

    def forward(self, x):
        all_feature_maps = []
        for i, encoder in enumerate(self.encoders):
            feature_maps = encoder(x[:, i : i + 1, ...])
            all_feature_maps.append(feature_maps)

        all_feature_maps = self.concat_features(all_feature_maps)
        feature_maps = self.decoder(all_feature_maps)
        return self.head(feature_maps[0])

    @classmethod
    def from_config(cls, config: DictConfig):
        from hydra.utils import instantiate

        encoders = []
        for _ in range(config.num_channels):
            encoder: EncoderModule = instantiate(config.encoder).change_input_channels(1)
            encoders.append(encoder)

        head = instantiate(config.head, channels=config.decoder.channels[0])
        box_coder = instantiate(
            config.box_coder,
            output_stride=encoders[0].strides[0] // (head.upsample_factor),
        )

        return cls(
            encoders=encoders,
            decoder_features=config.decoder.channels,
            head=head,
            box_coder=box_coder,
            activation=config.activation,
            block_type=config.decoder.get("block_type", "UnetBlock"),
            upsample_type=config.decoder.get("upsample_type", "bilinear"),
        )
