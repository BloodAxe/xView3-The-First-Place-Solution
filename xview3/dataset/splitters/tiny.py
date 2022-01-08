import abc
import os
from abc import abstractmethod
from functools import partial
from typing import Tuple

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .base import DatasetSplitter


__all__ = [
    "TinyDatasetSplitter",
]


class TinyDatasetSplitter(DatasetSplitter):
    def __init__(self, data_dir):
        super().__init__(data_dir, "tiny")

    def train_test_split(self, fold: int, num_folds: int):
        if (fold is not None and fold != 0) or num_folds is not None:
            raise RuntimeError("Tiny dataset does not support fold split")

        train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        train_df["scene_path"] = train_df["scene_id"].apply(partial(self.append_prefix, folder="train"))
        train_df["location"] = train_df["scene_id"]
        train_df["folder"] = "train"

        valid_df = pd.read_csv(os.path.join(self.data_dir, "validation.csv"))
        valid_df["scene_path"] = valid_df["scene_id"].apply(partial(self.append_prefix, folder="validation"))
        valid_df["location"] = valid_df["scene_id"]
        valid_df["folder"] = "validation"

        train_df = train_df[
            train_df.scene_id.isin(
                {
                    "05bc615a9b0e1159t",
                    "72dba3e82f782f67t",
                    "2899cfb18883251bt",
                    "e98ca5aba8849b06t",
                    "cbe4ad26fe73f118t",
                }
            )
        ].reset_index(drop=True)

        valid_df = valid_df[valid_df.scene_id.isin({"590dd08f71056cacv", "b1844cde847a3942v"})].reset_index(drop=True)

        shore_root = os.path.join(self.data_dir, self.dataset_size)
        return train_df, valid_df, valid_df, shore_root
