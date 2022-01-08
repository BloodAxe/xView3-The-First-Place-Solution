import torch.nn.init
from pytorch_toolbelt.modules import (
    conv1x1,
    instantiate_activation_block,
    ResidualDeconvolutionUpsample2d,
    ACT_RELU,
)
from torch import nn

__all__ = ["S4Head"]

from xview3.centernet.constants import (
    CENTERNET_OUTPUT_FISHING_MAP,
    CENTERNET_OUTPUT_VESSEL_MAP,
    CENTERNET_OUTPUT_OBJECTNESS_MAP,
    CENTERNET_OUTPUT_SIZE,
)
from xview3.centernet.models.heads.length import LengthParametrization


def conv_bn_act(input_channels, output_channels, activation):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        instantiate_activation_block(activation, inplace=True),
    )


class S4Head(nn.Module):
    upsample_factor: int = 4

    def __init__(
        self,
        channels: int,
        classifier_dim: int,
        dropout_rate=0.25,
        objectness_dim: int = 32,
        size_dim: int = 32,
        activation: str = ACT_RELU,
    ):
        super().__init__()
        self.objectness = nn.Sequential(
            # Conv Blocks
            conv_bn_act(channels, objectness_dim, activation),
            conv_bn_act(objectness_dim, objectness_dim, activation),
            # S4 -> S2
            ResidualDeconvolutionUpsample2d(objectness_dim),
            # Conv Blocks
            conv_bn_act(objectness_dim // 4, objectness_dim // 2, activation),
            conv_bn_act(objectness_dim // 2, objectness_dim // 2, activation),
            # Logits
            nn.Dropout2d(dropout_rate, inplace=False),
            conv1x1(objectness_dim // 2, 4),
            nn.PixelShuffle(2),
        )

        self.classifier = nn.Sequential(
            # Conv Blocks
            conv_bn_act(channels, classifier_dim, activation),
            conv_bn_act(objectness_dim, classifier_dim // 2, activation),
            # S4 -> S2
            nn.UpsamplingBilinear2d(scale_factor=2),
            # Conv Blocks
            conv_bn_act(classifier_dim // 2, classifier_dim // 2, activation),
            conv_bn_act(classifier_dim // 2, classifier_dim // 2, activation),
            # Logits
            nn.Dropout2d(dropout_rate, inplace=False),
            conv1x1(classifier_dim // 2, 2),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.size = nn.Sequential(
            # Conv Blocks
            conv_bn_act(channels, size_dim, activation),
            conv_bn_act(size_dim, size_dim // 2, activation),
            # S4 -> S2
            nn.UpsamplingBilinear2d(scale_factor=2),
            # Conv Blocks
            conv_bn_act(size_dim // 2, size_dim // 2, activation),
            conv_bn_act(size_dim // 2, size_dim // 2, activation),
            # Logits
            nn.Dropout2d(dropout_rate, inplace=False),
            conv1x1(size_dim // 2, 1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            LengthParametrization(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # torch.nn.init.constant_(self.objectness.final[-1].bias, -3)

    def forward(self, features):
        classifier = self.classifier(features)
        output = {
            CENTERNET_OUTPUT_OBJECTNESS_MAP: self.objectness(features),
            CENTERNET_OUTPUT_VESSEL_MAP: classifier[:, 0:1, ...],
            CENTERNET_OUTPUT_FISHING_MAP: classifier[:, 1:2, ...],
            CENTERNET_OUTPUT_SIZE: self.size(features),
        }

        return output
