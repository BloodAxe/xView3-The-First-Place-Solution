import os
from functools import partial

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

__all__ = [
    "FullDatasetWithHoldoutSplitter",
]
from .base import DatasetSplitter
from .validation import ValidationOnlySplitter


class FullDatasetWithHoldoutSplitter(ValidationOnlySplitter):
    """
    Split full train & test into proper folds & holdout
    """

    def __init__(self, data_dir: str):
        super().__init__(data_dir)

    def get_full_train_df(self):
        df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        df["scene_path"] = df["scene_id"].apply(partial(self.append_prefix, folder="train"))
        df["location"] = None
        df["folder"] = "train"

        scene_split = pd.read_csv("configs/dataset/scene_split.csv")
        for scene_id in df.scene_id.unique():
            location = scene_split.loc[scene_split.scene_id == scene_id, "location"].values[0]
            if location == "19":
                # This is the very dirty hack to remove location 19, which is a "sink"
                # for all open-sea, shore-less scenes
                location = scene_id
            df.loc[df.scene_id == scene_id, "location"] = location

        return df

    def get_train_df_except_holdout(self, holdout_df):
        full_train_df = self.get_full_train_df()

        # Remove scenes that belong to holdout split
        full_train_df = full_train_df[~full_train_df.scene_id.isin(holdout_df.scene_id.unique())]
        return full_train_df.reset_index()

    def train_test_split(self, fold: int, num_folds: int):
        if num_folds <= 0:
            raise RuntimeError(f"Num folds must be positive. Received {num_folds}")

        holdout_df = self.get_holdout_df()

        main_training_data = self.get_full_validation_df()
        main_training_data["fold"] = -1

        skf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=42)
        for fold_index, (_, valid_idx) in enumerate(
            skf.split(main_training_data, y=main_training_data.distance_from_shore_km < 2, groups=main_training_data.scene_id)
        ):
            main_training_data.loc[valid_idx, "fold"] = fold_index

        extra_training_data = self.get_train_df_except_holdout(holdout_df)
        extra_training_data["fold"] = -1
        for fold_index, (_, valid_idx) in enumerate(
            skf.split(extra_training_data, y=extra_training_data.is_fishing.fillna(2).astype(int), groups=extra_training_data.scene_id)
        ):
            extra_training_data.loc[valid_idx, "fold"] = fold_index

        shore_root = os.path.join(self.data_dir, self.dataset_size)
        train_df = pd.concat(
            [extra_training_data[extra_training_data.fold != fold], main_training_data[main_training_data.fold != fold]]
        ).reset_index(drop=True)
        valid_df = main_training_data[main_training_data.fold == fold].reset_index(drop=True)

        return train_df, valid_df, holdout_df, shore_root
