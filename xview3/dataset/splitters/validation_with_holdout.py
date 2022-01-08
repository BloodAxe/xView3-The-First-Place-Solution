import os
from functools import partial

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

__all__ = [
    "ValidationOnlyWithHoldoutSplitter",
]


class ValidationOnlyWithHoldoutSplitter:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.dataset_size = "full"

    def train_test_split(self, fold: int, num_folds: int):
        if num_folds <= 0:
            raise RuntimeError(f"Num folds must be positive. Received {num_folds}")

        def append_prefix(x, folder):
            return os.path.join(self.data_dir, self.dataset_size, folder, x)

        df = pd.read_csv(os.path.join(self.data_dir, "validation.csv"))
        df["scene_path"] = df["scene_id"].apply(partial(append_prefix, folder="validation"))

        df["location"] = None
        scene_split = pd.read_csv("configs/dataset/scene_split.csv")
        for scene_id in df.scene_id.unique():
            location = scene_split.loc[scene_split.scene_id == scene_id, "location"].values[0]
            df.loc[df.scene_id == scene_id, "location"] = location

        df["fold"] = -1
        skf = StratifiedGroupKFold(n_splits=num_folds + 1, shuffle=True, random_state=42)
        for fold_index, (_, valid_idx) in enumerate(skf.split(df, y=df.distance_from_shore_km < 2, groups=df.scene_id)):
            df.loc[valid_idx, "fold"] = fold_index

        shore_root = os.path.join(self.data_dir, self.dataset_size)

        holdout_fold = num_folds
        train_df = df[(df.fold != fold) & (df.fold != holdout_fold)].reset_index(drop=True)
        valid_df = df[df.fold == fold].reset_index(drop=True)
        holdout_df = df[df.fold == holdout_fold].reset_index(drop=True)
        return train_df, valid_df, holdout_df, shore_root
