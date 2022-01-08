import torch
from pytorch_toolbelt.modules import ACT_RELU, get_activation_block
from torch import nn

from .length import LengthParametrization
from ...constants import (
    CENTERNET_OUTPUT_CLASS_MAP,
    CENTERNET_OUTPUT_OBJECTNESS_MAP,
    CENTERNET_OUTPUT_SIZE,
    CENTERNET_OUTPUT_OFFSET,
)

__all__ = [
    "CenterNetHeadV2",
    "CenterNetHeadV2WithObjectness",
]


class CenterNetHeadV2(nn.Module):
    upsample_factor: int = 1

    def __init__(
        self,
        channels: int,
        embedding_size: int,
        num_classes: int,
        dropout_rate=0.25,
        activation=ACT_RELU,
        inplace=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout2d(dropout_rate, inplace=inplace)

        activation_block = get_activation_block(activation)

        self.heatmap_tail = nn.Sequential(
            nn.Conv2d(channels, embedding_size, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(embedding_size),
            activation_block(inplace=True),
        )

        self.class_head = nn.Conv2d(embedding_size, num_classes, kernel_size=1)
        self.class_head.bias.data.fill_(-1)

        self.size_head = nn.Sequential(
            nn.Conv2d(channels, embedding_size, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(embedding_size),
            activation_block(inplace=True),
            nn.Conv2d(embedding_size, 1, kernel_size=1),
            LengthParametrization(),
        )

        self.offset_head = nn.Sequential(
            nn.Conv2d(channels, embedding_size, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(embedding_size),
            activation_block(inplace=True),
            nn.Conv2d(embedding_size, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor):
        features = self.dropout(features)

        features_tail = self.heatmap_tail(features)
        output = {
            CENTERNET_OUTPUT_CLASS_MAP: self.class_head(features_tail),
            CENTERNET_OUTPUT_SIZE: self.size_head(features),
            CENTERNET_OUTPUT_OFFSET: self.offset_head(features),
        }
        return output


class CenterNetHeadV2WithObjectness(CenterNetHeadV2):
    def __init__(
        self,
        channels: int,
        embedding_size: int,
        num_classes: int,
        dropout_rate=0.25,
        activation=ACT_RELU,
        inplace=True,
    ):
        super().__init__(
            channels=channels,
            embedding_size=embedding_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            activation=activation,
            inplace=inplace,
        )
        self.class_head.bias.data.fill_(0)

        self.objectness_head = nn.Conv2d(embedding_size, 1, kernel_size=1)
        self.objectness_head.bias.data.fill_(-2.19)

    def forward(self, features: torch.Tensor):
        features = self.dropout(features)

        features_tail = self.heatmap_tail(features)
        output = {
            CENTERNET_OUTPUT_OBJECTNESS_MAP: self.objectness_head(features_tail),
            CENTERNET_OUTPUT_CLASS_MAP: self.class_head(features_tail),
            CENTERNET_OUTPUT_SIZE: self.size_head(features),
            CENTERNET_OUTPUT_OFFSET: self.offset_head(features),
        }
        return output
