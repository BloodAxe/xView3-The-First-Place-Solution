from collections import OrderedDict
from typing import Dict, Tuple, List

from pytorch_toolbelt.modules import ACT_RELU, get_activation_block, conv1x1
from torch import nn, Tensor

from ...constants import (
    CENTERNET_OUTPUT_VESSEL_MAP,
    CENTERNET_OUTPUT_FISHING_MAP,
    CENTERNET_OUTPUT_OBJECTNESS_MAP,
    CENTERNET_OUTPUT_SIZE,
    CENTERNET_OUTPUT_OFFSET,
)

__all__ = ["DecoupledHeadGroupNormLateShuffle"]


class FeatureRecombination(nn.Module):
    def __init__(self, channels, modulating_features):
        super().__init__()
        self.project = conv1x1(modulating_features, channels)

    def forward(self, x, modulation):
        return x * self.project(modulation).sigmoid()


def build_regression_block(
    input_channels: int,
    embedding_size: int,
    dropout_rate: float,
    num_blocks: int,
    activation: str,
    extra_final_layers: List[Tuple[str, nn.Module]],
    num_groups=16,
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
            (f"bn{layer_index}", nn.GroupNorm(num_groups, embedding_size)),
            (f"act{layer_index}", activation_block(inplace=True)),
        ]
    layers += [
        ("drop", nn.Dropout2d(dropout_rate, inplace=False)),
        ("conv1", nn.Conv2d(embedding_size, embedding_size, kernel_size=(1, 1), padding=0, bias=False)),
    ] + extra_final_layers

    return nn.Sequential(OrderedDict(layers))


class DecoupledHeadGroupNormLateShuffle(nn.Module):
    upsample_factor: int = 2

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
        num_groups=16,
    ):
        super().__init__()
        self.objectness = build_regression_block(
            channels,
            embedding_size=objectness_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            num_groups=num_groups,
            num_blocks=num_blocks,
            extra_final_layers=[
                (("shuffle"), nn.PixelShuffle(2)),
                ("final", nn.Conv2d(objectness_dim // 4, 1, kernel_size=(3, 3), padding=1)),
            ],
        )

        self.classifier = build_regression_block(
            channels,
            embedding_size=classifier_dim,
            dropout_rate=dropout_rate,
            num_groups=num_groups,
            activation=activation,
            num_blocks=num_blocks,
            extra_final_layers=[],
        )

        self.size = build_regression_block(
            channels,
            embedding_size=size_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            num_groups=num_groups,
            num_blocks=num_blocks,
            extra_final_layers=[],
        )

        self.classifier_recombination = FeatureRecombination(classifier_dim, size_dim)
        self.classifier_up = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(classifier_dim, classifier_dim, kernel_size=(3, 3), padding=1)),
                    ("shuffle", nn.PixelShuffle(2)),
                    ("final", nn.Conv2d(classifier_dim // 4, 2, kernel_size=(3, 3), padding=1)),
                ]
            )
        )
        self.size_recombination = FeatureRecombination(size_dim, classifier_dim)
        self.size_up = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(size_dim, size_dim, kernel_size=(3, 3), padding=1)),
                    ("shuffle", nn.PixelShuffle(2)),
                    ("final", nn.Conv2d(size_dim // 4, 1, kernel_size=(3, 3), padding=1)),
                    ("activation", nn.Softplus()),
                ]
            )
        )

        if offset_dim:
            self.offset = build_regression_block(
                channels,
                embedding_size=offset_dim,
                dropout_rate=dropout_rate,
                activation=activation,
                num_groups=num_groups,
                num_blocks=num_blocks,
                extra_final_layers=[
                    (("shuffle"), nn.PixelShuffle(2)),
                    ("final", nn.Conv2d(offset_dim // 4, 2, kernel_size=(3, 3), padding=1)),
                ],
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

        # self.classifier_up.final.bias.data.fill_(0)
        # self.objectness.final.bias.data.fill_(-4)

        self.objectness.final.bias.data.fill_(-5)
        self.classifier_up.final.bias.data.fill_(-1)
        self.size_up.final.bias.data.fill_(2)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        class_features = self.classifier(features)
        length_features = self.size(features)

        new_class_features = self.classifier_recombination(class_features, length_features)
        new_length_features = self.size_recombination(length_features, class_features)

        predicted_classes = self.classifier_up(new_class_features)
        predicted_length = self.size_up(new_length_features)

        output = {
            CENTERNET_OUTPUT_OBJECTNESS_MAP: self.objectness(features),
            CENTERNET_OUTPUT_VESSEL_MAP: predicted_classes[:, 0:1],
            CENTERNET_OUTPUT_FISHING_MAP: predicted_classes[:, 1:2],
            CENTERNET_OUTPUT_SIZE: predicted_length,
        }
        if self.offset is not None:
            output[CENTERNET_OUTPUT_OFFSET] = self.offset(features).clip(0, 1)
        return output
