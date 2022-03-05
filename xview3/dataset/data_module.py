import os
from collections import namedtuple
from typing import Tuple, Optional, Union, Dict

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_toolbelt.utils import to_numpy

__all__ = [
    "XView3DataModule",
    "ignore_low_confidence_objects",
    "filter_low_confidence_objects",
    "drop_unused_columns",
    "MultilabelTargets",
]

from xview3.constants import PIX_TO_M, IGNORE_LABEL

from .splitters import *


def ignore_low_confidence_objects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set is_vessel, is_fishing and vessel_length_m fields to nan for low-confidence objects.

    Args:
        df:

    Returns:

    """
    df = df.copy()
    low_confidence_mask = df.confidence == "LOW"
    df.loc[low_confidence_mask, "is_vessel"] = float("nan")
    df.loc[low_confidence_mask, "is_fishing"] = float("nan")
    df.loc[low_confidence_mask, "vessel_length_m"] = float("nan")
    return df


def drop_unused_columns(df):
    return df.drop(columns=["detect_lat", "detect_lon", "top", "left", "bottom", "right"]).copy().reset_index(drop=True)


def filter_low_confidence_objects(df):
    return df[df["confidence"].isin(["HIGH", "MEDIUM"])].copy().reset_index(drop=True)


MultilabelTargets = namedtuple(
    "MultilabelTargets",
    (
        "centers",
        "confidences",
        "is_vessel",
        "is_fishing",
        "lengths",
        "is_near_shore",
        "is_vessel_probs",
        "is_fishing_probs",
        "objectness_probs"
    ),
)


class XView3DataModule:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    @classmethod
    def get_label_targets_from_df(cls, df: pd.DataFrame):
        centers = np.stack([df.detect_scene_column.values, df.detect_scene_row.values], axis=-1)

        labels = cls.encode_labels(df.is_vessel, df.is_fishing)
        lengths = cls.encode_lengths(df.vessel_length_m)

        return centers.reshape((-1, 2)), labels.reshape((-1)), lengths.reshape((-1))

    @classmethod
    def get_multilabel_targets_from_df(cls, df: pd.DataFrame) -> MultilabelTargets:
        centers = np.stack([df.detect_scene_column.values, df.detect_scene_row.values], axis=-1)
        if "confidence" in df:
            confidences = df.confidence.values.copy().reshape((-1))
        else:
            confidences = None

        if "distance_from_shore_km" in df:
            is_near_shore = df.distance_from_shore_km.values <= 2
            is_near_shore = is_near_shore.reshape((-1)).astype(int)
        else:
            is_near_shore = None
        lengths = cls.encode_lengths(df.vessel_length_m)

        is_vessel = df.is_vessel.values.astype(np.float32)
        is_vessel[~np.isfinite(is_vessel)] = IGNORE_LABEL

        is_fishing = df.is_fishing.values.astype(np.float32)
        is_fishing[~np.isfinite(is_fishing)] = IGNORE_LABEL
        # is_fishing[is_vessel == 0] = 0 # If it is non-vessel, then by definition it cannot do fishing

        if "objectness_p" in df:
            objectness_p = df.objectness_p.values
        else:
            objectness_p = None

        if "is_vessel_p" in df:
            is_vessel_p = df.is_vessel_p.values
        else:
            is_vessel_p = None

        if "is_fishing_p" in df:
            is_fishing_p = df.is_fishing_p.values
        else:
            is_fishing_p = None


        return MultilabelTargets(
            centers=centers.reshape((-1, 2)),
            is_vessel=is_vessel.reshape((-1)).astype(int),
            is_fishing=is_fishing.reshape((-1)).astype(int),
            lengths=lengths.reshape((-1)),
            confidences=confidences,
            is_near_shore=is_near_shore,
            is_fishing_probs=is_fishing_p,
            is_vessel_probs=is_vessel_p,
            objectness_probs=objectness_p
        )

    @classmethod
    def encode_lengths(cls, lengths: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Convert length in meters to length in pixels
        :param lengths:
        :return:
        """
        lengths = np.asarray(lengths).astype(dtype=np.float32, copy=True)
        invalid_lengths = ~np.isfinite(lengths)
        lengths[invalid_lengths] = 0
        return lengths / PIX_TO_M

    @classmethod
    def decode_lengths(cls, lengths: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        return to_numpy(lengths * PIX_TO_M)

    @classmethod
    def encode_labels(cls, is_vessel: Union[np.ndarray, pd.Series], is_fishing: Union[np.ndarray, pd.Series]) -> np.ndarray:
        is_fishing_vec = np.asarray(is_fishing == True).astype(int)
        is_vessel_vec = np.asarray((is_fishing == True) | (is_fishing == False) | (is_vessel == True)).astype(int)
        labels = is_vessel_vec + is_fishing_vec
        return labels

    @classmethod
    def decode_labels(cls, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        is_fishing = labels == 2
        is_vessel = labels != 0
        return is_vessel, is_fishing

    @classmethod
    def iterate_crops(cls, image_size: Tuple[int, int], tile_size: Tuple[int, int], tile_step: Tuple[int, int]):
        image_height, image_width = image_size[:2]
        for y in range(0, image_height - tile_size[0] + 1, tile_step[0]):
            for x in range(0, image_width - tile_size[1] + 1, tile_step[1]):
                yield (y, y + tile_step[0]), (x, x + tile_step[1])

    def train_val_split(
        self, splitter: Union[str, Dict, DictConfig], fold: Optional[int] = None, num_folds: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        if isinstance(splitter, str):
            splitter_name = splitter
            splitter_params = {}
        else:
            splitter_name = splitter["name"]
            splitter_params = dict((k, v) for k, v in splitter.items() if k != "name")

        splitter_cls: DatasetSplitter = {
            "precomputed": PrecomputedSplitter,
            "tiny": TinyDatasetSplitter,
            "valid_only": ValidationOnlySplitter,
            "valid_only_with_holdout": ValidationOnlyWithHoldoutSplitter,
            "full": FullDatasetWithHoldoutSplitter,
        }[splitter_name](data_dir=self.data_dir, **splitter_params)
        return splitter_cls.train_test_split(fold=fold, num_folds=num_folds)

    def get_test_scenes(self):
        test_dir = os.path.join(self.data_dir, "test")
        return [os.path.join(test_dir, scene) for scene in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, scene))]
