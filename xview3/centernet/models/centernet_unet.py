from functools import partial
from typing import List

from omegaconf import DictConfig
from pytorch_toolbelt.modules import (
    ACT_RELU,
    decoders as D,
    EncoderModule,
    ResidualDeconvolutionUpsample2d,
)
from torch import nn

from .unet_blocks import IRBlock, get_unet_block

__all__ = ["CenterNetUNetModel"]


class CenterNetUNetModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        decoder_features: List[int],
        head: nn.Module,
        box_coder,
        input_upsample_factor: int = 1,
        activation=ACT_RELU,
        num_extra_blocks=0,
        block_type="UnetBlock",
        upsample_type="bilinear",
        extra_block=partial(IRBlock, stride=2),
    ):
        super().__init__()
        self.output_stride = encoder.strides[0]
        self.initial_upsample = (
            nn.UpsamplingBilinear2d(scale_factor=input_upsample_factor) if input_upsample_factor != 1 else nn.Identity()
        )

        unet_block = get_unet_block(block_type, activation=activation)
        self.encoder = encoder
        self.extra_encoder_blocks = nn.ModuleList(
            [extra_block(encoder.channels[-1], encoder.channels[-1]) for _ in range(num_extra_blocks)]
        )

        upsample_block = {
            "nearest": nn.UpsamplingNearest2d,
            "bilinear": nn.UpsamplingBilinear2d,
            "rdtsc": ResidualDeconvolutionUpsample2d,
            "shuffle": nn.PixelShuffle,
        }[upsample_type]

        encoder_channels = list(encoder.channels) + [encoder.channels[-1]] * num_extra_blocks
        self.decoder = D.UNetDecoder(
            encoder_channels,
            decoder_features,
            unet_block=unet_block,
            upsample_block=upsample_block,
        )
        self.head = head
        self.box_coder = box_coder

    def forward(self, x):
        x = self.initial_upsample(x)
        feature_maps = self.encoder(x)
        # Add extra feature maps
        for extra_block in self.extra_encoder_blocks:
            feature_maps.append(extra_block(feature_maps[-1]))

        feature_maps = self.decoder(feature_maps)
        return self.head(feature_maps[0])

    @classmethod
    def from_config(cls, config: DictConfig):
        from hydra.utils import instantiate

        encoder: EncoderModule = instantiate(config.encoder)
        if config.num_channels != 3:
            encoder = encoder.change_input_channels(config.num_channels)

        input_upsample_factor: int = int(config.get("input_upsample_factor", 1))

        head = instantiate(config.head, channels=config.decoder.channels[0])
        box_coder = instantiate(
            config.box_coder,
            output_stride=encoder.strides[0] // (head.upsample_factor * input_upsample_factor),
        )

        return CenterNetUNetModel(
            encoder=encoder,
            decoder_features=config.decoder.channels,
            head=head,
            input_upsample_factor=input_upsample_factor,
            box_coder=box_coder,
            activation=config.activation,
            block_type=config.decoder.get("block_type", "UnetBlock"),
            upsample_type=config.decoder.get("upsample_type", "bilinear"),
            num_extra_blocks=config.num_extra_blocks,
        )
