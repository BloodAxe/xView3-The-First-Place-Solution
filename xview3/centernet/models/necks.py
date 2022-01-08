import collections
from typing import List, Dict
from pytorch_toolbelt.modules import FPNFuse
from torch import nn, Tensor

__all__ = ["TensorAtIndex", "PickTensors", "HyperColumnNeck"]


class TensorAtIndex(nn.Module):
    def __init__(self, channels: List[int], index: int = 0):
        super().__init__()
        self.index = index
        self.channels = channels[index]

    def forward(self, feature_maps: List[Tensor]) -> Tensor:
        return feature_maps[self.index]


class PickTensors(nn.Module):
    def __init__(self, channels: List[int], indexes: Dict[int, str]):
        super().__init__()
        self.indexes = indexes
        self.channels = channels[0]

    def forward(self, feature_maps: List[Tensor]) -> Dict[str, Tensor]:
        return collections.OrderedDict((name, feature_maps[index]) for index, name in self.indexes.items())


class HyperColumnNeck(nn.Module):
    def __init__(self, channels: List[int], dropout_rate=0.0):
        super().__init__()
        self.fuse = FPNFuse(align_corners=True)
        self.channels = sum(channels)
        self.dropout = nn.Dropout2d(p=dropout_rate, inplace=True)

    def forward(self, feature_maps: List[Tensor]) -> Tensor:
        return self.dropout(self.fuse(feature_maps))
