from typing import List

from omegaconf import DictConfig
from pytorch_toolbelt.modules import (
    ACT_RELU,
    decoders as D,
    EncoderModule,
    ResidualDeconvolutionUpsample2d,
)
from torch import nn

from .unet_blocks import get_unet_block

__all__ = ["CenterNetBiFPNUNetModel"]

from .decoders import BiFPN


class CenterNetBiFPNUNetModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        decoder_features: List[int],
        head: nn.Module,
        box_coder,
        bifpn_channels=256,
        bifpn_layers=3,
        activation=ACT_RELU,
        block_type="UnetBlock",
        upsample_type: str = "bilinear",
    ):
        super().__init__()
        self.output_stride = encoder.strides[0]

        self.encoder = encoder
        self.transition = BiFPN(
            self.encoder.channels, self.encoder.strides, channels=bifpn_channels, num_layers=bifpn_layers, activation=activation
        )

        upsample_block = {
            "nearest": nn.UpsamplingNearest2d,
            "bilinear": nn.UpsamplingBilinear2d,
            "rdtsc": ResidualDeconvolutionUpsample2d,
            "shuffle": nn.PixelShuffle,
        }[upsample_type]

        decoder_input_channels = [bifpn_channels] * 5
        unet_block = get_unet_block(block_type, activation=activation)
        self.decoder = D.UNetDecoder(
            decoder_input_channels,
            decoder_features,
            unet_block=unet_block,
            upsample_block=upsample_block,
        )
        self.head = head
        self.box_coder = box_coder
        self._init_weights(self.transition)
        self._init_weights(self.decoder)

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feature_maps = self.encoder(x)
        feature_maps = self.transition(feature_maps)
        feature_maps = self.decoder(feature_maps)
        return self.head(feature_maps[0])

    @classmethod
    def from_config(cls, config: DictConfig):
        from hydra.utils import instantiate

        encoder: EncoderModule = instantiate(config.encoder)
        if config.num_channels != 3:
            encoder = encoder.change_input_channels(config.num_channels)

        head = instantiate(config.head, channels=config.decoder.channels[0])
        box_coder = instantiate(
            config.box_coder,
            output_stride=encoder.strides[0] // head.upsample_factor,
        )

        return cls(
            encoder=encoder,
            decoder_features=config.decoder.channels,
            head=head,
            box_coder=box_coder,
            bifpn_layers=config.bifpn.num_layers,
            bifpn_channels=config.bifpn.channels,
            activation=config.activation,
            block_type=config.decoder.get("block_type", "UnetBlock"),
            upsample_type=config.decoder.get("upsample_type", "bilinear"),
        )
