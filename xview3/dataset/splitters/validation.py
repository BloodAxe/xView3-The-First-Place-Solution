import os
from functools import partial

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

__all__ = [
    "ValidationOnlySplitter",
]


class ValidationOnlySplitter:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.dataset_size = "full"

    def append_prefix(self, x, folder):
        return os.path.join(self.data_dir, self.dataset_size, folder, x)

    def get_holdout_df(self):
        """
        As a holdout we use since scene from each train
        Returns:

        """
        df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        df["location"] = None
        df["folder"] = "train"

        scene_split = pd.read_csv("configs/dataset/scene_split.csv")
        for scene_id in df.scene_id.unique():
            location = scene_split.loc[scene_split.scene_id == scene_id, "location"].values[0]
            df.loc[df.scene_id == scene_id, "location"] = location

        scenes = []
        for location_id in df.location.unique():
            location_df = df[df.location == location_id]
            scene_id = location_df.scene_id.unique()[0]
            scene_df = location_df[location_df.scene_id == scene_id]
            scenes.append(scene_df)

        holdout_df = pd.concat(scenes)
        holdout_df["scene_path"] = holdout_df["scene_id"].apply(partial(self.append_prefix, folder="train"))
        return holdout_df.reset_index()

    def get_full_validation_df(self):
        df = pd.read_csv(os.path.join(self.data_dir, "validation.csv"))
        df["scene_path"] = df["scene_id"].apply(partial(self.append_prefix, folder="validation"))
        df["location"] = None
        df["folder"] = "validation"

        scene_split = pd.read_csv("configs/dataset/scene_split.csv")
        for scene_id in df.scene_id.unique():
            location = scene_split.loc[scene_split.scene_id == scene_id, "location"].values[0]
            df.loc[df.scene_id == scene_id, "location"] = location
        return df

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

        shore_root = os.path.join(self.data_dir, self.dataset_size)

        train_df = main_training_data[main_training_data.fold != fold].reset_index(drop=True)
        valid_df = main_training_data[main_training_data.fold == fold].reset_index(drop=True)

        return train_df, valid_df, holdout_df, shore_root
