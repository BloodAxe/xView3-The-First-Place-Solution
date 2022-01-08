from typing import Dict

from pytorch_toolbelt.modules import ACT_RELU
from torch import nn, Tensor

from .deep_head import build_deep_block, DeconvolutionUpsample2d
from .length import LengthParametrization
from ...constants import (
    CENTERNET_OUTPUT_VESSEL_MAP,
    CENTERNET_OUTPUT_FISHING_MAP,
    CENTERNET_OUTPUT_OBJECTNESS_MAP,
    CENTERNET_OUTPUT_SIZE,
    CENTERNET_OUTPUT_OFFSET,
)

__all__ = ["MultilabelHydraHead"]


class MultilabelHydraHead(nn.Module):
    upsample_factor: int

    def __init__(
        self,
        channels: int,
        classifier_dim: int,
        dropout_rate=0.25,
        objectness_dim: int = 32,
        size_dim: int = 32,
        offset_dim: int = 32,
        num_blocks: int = 3,
        activation: str = ACT_RELU,
        upsample_factor: int = 2,
    ):
        super().__init__()
        self.upsample_factor = upsample_factor
        mid_channels = channels // upsample_factor

        self.up = DeconvolutionUpsample2d(channels, mid_channels) if upsample_factor == 2 else nn.Identity()

        self.is_vessel = build_deep_block(
            mid_channels,
            1,
            embedding_size=classifier_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            num_blocks=num_blocks,
            extra_final_layers=[],
        )
        self.is_fishing = build_deep_block(
            mid_channels,
            1,
            embedding_size=classifier_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            num_blocks=num_blocks,
            extra_final_layers=[],
        )
        self.objectness = build_deep_block(
            mid_channels,
            1,
            embedding_size=objectness_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            num_blocks=num_blocks,
            extra_final_layers=[],
        )
        self.size = build_deep_block(
            mid_channels,
            1,
            embedding_size=size_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            num_blocks=num_blocks,
            extra_final_layers=[("softplus", nn.Softplus())],
        )

        if offset_dim:
            self.offset = build_deep_block(
                mid_channels,
                2,
                embedding_size=offset_dim,
                dropout_rate=dropout_rate,
                activation=activation,
                num_blocks=num_blocks,
                extra_final_layers=[],
            )
        else:
            self.offset = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.is_vessel.final.bias.data.fill_(0)
        self.is_fishing.final.bias.data.fill_(0)
        self.objectness.final.bias.data.fill_(-2.618)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        features = self.up(features)
        output = {
            CENTERNET_OUTPUT_VESSEL_MAP: self.is_vessel(features),
            CENTERNET_OUTPUT_FISHING_MAP: self.is_fishing(features),
            CENTERNET_OUTPUT_OBJECTNESS_MAP: self.objectness(features),
            CENTERNET_OUTPUT_SIZE: self.size(features),
        }
        if self.offset is not None:
            output[CENTERNET_OUTPUT_OFFSET] = self.offset(features).clip(0, 1)
        return output
