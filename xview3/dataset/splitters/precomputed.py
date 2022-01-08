import os
from functools import partial

import pandas as pd

__all__ = [
    "PrecomputedSplitter",
]


class PrecomputedSplitter:
    def __init__(self, data_dir: str, split_csv: str):
        self.data_dir = data_dir
        self.dataset_size = "full"
        self.split_csv = split_csv

    def train_test_split(self, fold: int, num_folds: int):
        if num_folds <= 0:
            raise RuntimeError(f"Num folds must be positive. Received {num_folds}")

        def append_prefix(x, folder):
            return os.path.join(self.data_dir, self.dataset_size, folder, x)

        df1 = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        df1["scene_path"] = df1["scene_id"].apply(partial(append_prefix, folder="train"))

        df2 = pd.read_csv(os.path.join(self.data_dir, "validation.csv"))
        df2["scene_path"] = df2["scene_id"].apply(partial(append_prefix, folder="validation"))

        df = pd.concat([df1, df2])
        df["fold"] = -1
        df["location"] = None
        # We take only scenes that exists in split_csv
        df_remaining = []
        # Assign folds from other CSV
        split = pd.read_csv(self.split_csv)
        for i, row in split.iterrows():
            df.loc[df.scene_id == row["scene_id"], "fold"] = row["fold"]
            df.loc[df.scene_id == row["scene_id"], "location"] = row["location"]
            df_remaining.append(df.loc[df.scene_id == row["scene_id"]])

        df = pd.concat(df_remaining)
        shore_root = os.path.join(self.data_dir, self.dataset_size)

        train_df = df[(df.fold >= 0) & (df.fold != fold)].reset_index(drop=True)
        valid_df = df[df.fold == fold].reset_index(drop=True)
        holdout_df = df[df.fold == -1].reset_index(drop=True)
        return train_df, valid_df, holdout_df, shore_root
