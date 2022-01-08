from typing import Dict

from pytorch_toolbelt.modules import ACT_RELU
from torch import nn, Tensor

from .deep_head import build_deep_block, DeconvolutionUpsample2d
from ...constants import (
    CENTERNET_OUTPUT_VESSEL_MAP,
    CENTERNET_OUTPUT_FISHING_MAP,
    CENTERNET_OUTPUT_OBJECTNESS_MAP,
    CENTERNET_OUTPUT_SIZE,
    CENTERNET_OUTPUT_OFFSET,
)

__all__ = ["MultilabelCenterNetDeepHead"]


class MultilabelCenterNetDeepHead(nn.Module):
    upsample_factor: int = 2

    def __init__(
        self,
        channels: int,
        classifier_dim: int,
        dropout_rate=0.25,
        objectness_dim: int = 32,
        size_dim: int = 32,
        offset_dim: int = None,
        num_blocks: int = 3,
        activation: str = ACT_RELU,
    ):
        super().__init__()
        if offset_dim is None:
            offset_dim = size_dim
        self.up = DeconvolutionUpsample2d(channels, channels // 2)
        self.classifier = build_deep_block(
            channels // 2,
            2,
            embedding_size=classifier_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            num_blocks=num_blocks,
            extra_final_layers=[],
        )
        self.objectness = build_deep_block(
            channels // 2,
            1,
            embedding_size=objectness_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            num_blocks=num_blocks,
            extra_final_layers=[],
        )
        self.size = build_deep_block(
            channels // 2,
            1,
            embedding_size=size_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            num_blocks=num_blocks,
            extra_final_layers=[("activation", nn.Softplus())],
        )

        if offset_dim > 0:
            self.offset = build_deep_block(
                channels // 2,
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

        self.classifier.final.bias.data.fill_(0)
        self.objectness.final.bias.data.fill_(-4)
        self.size.final.bias.data.fill_(0)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        features = self.up(features)
        classifier = self.classifier(features)
        output = {
            CENTERNET_OUTPUT_VESSEL_MAP: classifier[:, 0:1, ...],
            CENTERNET_OUTPUT_FISHING_MAP: classifier[:, 1:2, ...],
            CENTERNET_OUTPUT_OBJECTNESS_MAP: self.objectness(features),
            CENTERNET_OUTPUT_SIZE: self.size(features),
        }

        if self.offset is not None:
            output[CENTERNET_OUTPUT_OFFSET] = self.offset(features).clip(0, 1)

        return output
