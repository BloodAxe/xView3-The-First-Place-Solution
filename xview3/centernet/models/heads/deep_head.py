from collections import OrderedDict
from typing import Dict, Tuple, List

from pytorch_toolbelt.modules import ACT_RELU, get_activation_block
from torch import nn, Tensor

from .length import LengthParametrization
from ...constants import (
    CENTERNET_OUTPUT_OBJECTNESS_MAP,
    CENTERNET_OUTPUT_SIZE,
    CENTERNET_OUTPUT_OFFSET,
    CENTERNET_OUTPUT_CLASS_MAP,
)

__all__ = ["CenterNetDeepHead"]


def build_deep_block(
    input_channels: int,
    output_channels: int,
    embedding_size: int,
    dropout_rate: float,
    num_blocks: int,
    activation: str,
    extra_final_layers: List[Tuple[str, nn.Module]],
):
    activation_block = get_activation_block(activation)

    layers = [
        (
            "project",
            nn.Conv2d(input_channels, embedding_size, kernel_size=(1, 1), padding=0, bias=False),
        ),
    ]

    for layer_index in range(num_blocks):
        layers += [
            (
                f"conv{layer_index}",
                nn.Conv2d(embedding_size, embedding_size, kernel_size=(3, 3), padding=1, bias=False),
            ),
            (f"bn{layer_index}", nn.BatchNorm2d(embedding_size)),
            (f"act{layer_index}", activation_block(inplace=True)),
        ]
    layers += [
        ("drop", nn.Dropout2d(dropout_rate, inplace=False)),
        ("final", nn.Conv2d(embedding_size, output_channels, kernel_size=(1, 1))),
    ] + extra_final_layers

    return nn.Sequential(OrderedDict(layers))


class DeconvolutionUpsample2d(nn.Module):
    def __init__(self, in_channels: int, n=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // 2
        self.conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, padding=1, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # skipcq: PYL-W0221
        output_size = x.size(2) * 2, x.size(3) * 2
        return self.conv(x, output_size=output_size)


class CenterNetDeepHead(nn.Module):
    upsample_factor: int = 2

    def __init__(
        self,
        channels: int,
        classifier_dim: int,
        num_classes: int,
        dropout_rate=0.25,
        objectness_dim: int = 32,
        size_dim: int = 32,
        num_blocks: int = 3,
        activation: str = ACT_RELU,
    ):
        super().__init__()

        self.up = DeconvolutionUpsample2d(channels, channels // 2)

        self.classification = build_deep_block(
            channels // 2,
            num_classes,
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
            extra_final_layers=[("length", LengthParametrization())],
        )
        self.offset = build_deep_block(
            channels // 2,
            2,
            embedding_size=size_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            num_blocks=num_blocks,
            extra_final_layers=[("sigmoid", nn.Sigmoid())],
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.classification.final.bias.data.fill_(-2.19)
        self.objectness.final.bias.data.fill_(-4)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        features = self.up(features)
        output = {
            CENTERNET_OUTPUT_CLASS_MAP: self.classification(features),
            CENTERNET_OUTPUT_OBJECTNESS_MAP: self.objectness(features),
            CENTERNET_OUTPUT_SIZE: self.size(features),
            CENTERNET_OUTPUT_OFFSET: self.offset(features),
        }
        return output
