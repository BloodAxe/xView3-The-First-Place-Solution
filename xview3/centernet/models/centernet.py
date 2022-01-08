from typing import Dict, Iterable

import torch
from omegaconf import DictConfig
from pytorch_toolbelt.modules import EncoderModule, DecoderModule
from torch import nn, Tensor

__all__ = ["CenterNetModel"]


class CenterNetModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        decoder: DecoderModule,
        head: nn.Module,
        box_coder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = head

        self.box_coder = box_coder

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)
        if not torch.is_tensor(dec_features):
            dec_features = dec_features[0]
        output = self.head(dec_features)
        return output

    @classmethod
    def from_config(cls, config: DictConfig):
        from hydra.utils import instantiate

        encoder = instantiate(config.encoder)
        if config.num_channels != 3:
            encoder = encoder.change_input_channels(config.num_channels)
        decoder = instantiate(config.decoder, feature_maps=encoder.channels)
        head = instantiate(config.head, channels=decoder.channels[0] if isinstance(decoder.channels, Iterable) else decoder.channels)
        box_coder = instantiate(config.box_coder, output_stride=encoder.strides[0] // head.upsample_factor)

        return CenterNetModel(encoder=encoder, decoder=decoder, head=head, box_coder=box_coder)
