import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from fire import Fire
from pytorch_toolbelt.utils import fs
from sklearn.model_selection import StratifiedGroupKFold

from xview3 import XView3DataModule

holdout = [
    "6a2b6ddecd398c6fv",  # validation_6a2b6ddecd398c6fv 13 Counter({'test': 7, 'train': 3, 'validation': 1})
    "3ceef682fbe4930av",  # validation_3ceef682fbe4930av 03 Counter({'train': 9, 'test': 6, 'validation': 1})
    "5c3d986db930f848v",  # validation_5c3d986db930f848v 10 Counter({'train': 6, 'test': 5, 'validation': 1})
    "128443d1e98e2839v",  # validation_128443d1e98e2839v 26 Counter({'test': 5, 'train': 4, 'validation': 1})
    "cdc04ca397865356v",  # validation_cdc04ca397865356v 37 Counter({'train': 2, 'test': 1, 'validation': 1})
    "b5272e098f7c7ff1t",  # train_b5272e098f7c7ff1t 44 Counter({'train': 1})
]


def main():
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)
    scene_split = defaultdict(list)

    groundtruths = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/validation.csv")])

    dirs = os.listdir("visualizations")
    for location in dirs:
        dir_path = os.path.join("visualizations", location)
        files = fs.find_in_dir(dir_path)
        splits = [fs.id_from_fname(x).split("_")[0] for x in files]
        scene_ids = [fs.id_from_fname(x).split("_")[1] for x in files]
        c = Counter(splits)
        for split, scene_id in zip(splits, scene_ids):
            scene_split["scene_id"].append(scene_id)
            scene_split["location"].append(location)
            scene_split["official_split"].append(split)
            #
            scene_split["scenes_count_in_train"].append(c.get("train", 0))
            scene_split["scenes_count_in_valid"].append(c.get("validation", 0))
            scene_split["scenes_count_in_test"].append(c.get("test", 0))
            scene_split["scenes_count_in_train_and_valid"].append(c.get("train", 0) + c.get("validation", 0))
            #
            scene_gts = groundtruths[groundtruths.scene_id == scene_id]

            is_vessel = scene_gts.is_vessel.values
            is_fishing = scene_gts.is_fishing.values

            scene_split["platform_instances"].append(sum(is_vessel == 0))
            scene_split["vessel_instances"].append(sum(is_vessel == 1))
            scene_split["fishing_instances"].append(sum(is_fishing == 1))
            scene_split["unknown_vessel"].append(np.sum(~np.isfinite(is_vessel.tolist()), dtype=int))
            scene_split["unknown_fishing"].append(np.sum(np.isfinite(is_vessel.tolist()) & ~np.isfinite(is_fishing.tolist()), dtype=int))

            recommended_split = split
            if len(c) == 1 and c.get("validation", 0) == 1 and split == "validation":
                recommended_split = "holdout"
            # if len(c) == 1 and len(scene_ids) <= 2 and "test" not in c:
            #     recommended_split = "holdout"
            if c.get("validation", 0) == 1 and split == "validation":
                recommended_split = "holdout"
            if len(c) == 1 and c.get("train", 0) > 0 and c.get("train", 0) <= 1 and split == "train":
                recommended_split = "holdout"

            scene_split["recommended_split"].append(recommended_split)
        print(location, c)

    scene_split = pd.DataFrame.from_dict(scene_split)
    scene_split["fold"] = -1

    print(
        scene_split[scene_split.official_split == "train"].sort_values(by="scenes_count_in_train_and_valid", ascending=True)[
            [
                "scene_id",
                "location",
                "official_split",
                "recommended_split",
                "scenes_count_in_train",
                "scenes_count_in_valid",
                "scenes_count_in_test",
                "fishing_instances",
            ]
        ]
    )
    exit(0)

    # print(scene_split)
    scene_split.to_csv("configs/dataset/scene_split.csv", index=False)
    print("Train      ", len(scene_split[scene_split.recommended_split == "train"]))
    print("Validation ", len(scene_split[scene_split.recommended_split == "validation"]))
    print("Holdout    ", len(scene_split[scene_split.recommended_split == "holdout"]))
    print("Test       ", len(scene_split[scene_split.recommended_split == "test"]))

    holdout = scene_split[scene_split.recommended_split == "holdout"]

    valid_data_except_holdout = (
        scene_split[(scene_split.official_split != "test") & (scene_split.recommended_split == "validation")].copy().reset_index(drop=True)
    )

    valid_data_except_holdout["average_platform_instances"] = 0
    valid_data_except_holdout["average_vessel_instances"] = 0
    valid_data_except_holdout["average_fishing_instances"] = 0

    for location in valid_data_except_holdout.location.unique():
        location_df = valid_data_except_holdout[valid_data_except_holdout.location == location]
        valid_data_except_holdout.loc[valid_data_except_holdout.location == location, "average_platform_instances"] = location_df[
            "platform_instances"
        ].mean()
        valid_data_except_holdout.loc[valid_data_except_holdout.location == location, "average_vessel_instances"] = location_df[
            "vessel_instances"
        ].mean()
        valid_data_except_holdout.loc[valid_data_except_holdout.location == location, "average_fishing_instances"] = location_df[
            "fishing_instances"
        ].mean()

    print(
        valid_data_except_holdout[
            [
                "scene_id",
                "location",
                "recommended_split",
                "fishing_instances",
                "average_fishing_instances",
                "scenes_count_in_train",
                "scenes_count_in_valid",
                "scenes_count_in_test",
            ]
        ].sort_values(by="average_fishing_instances")
    )

    skf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=2)
    for fold_index, (_, valid_idx) in enumerate(
        skf.split(
            valid_data_except_holdout,
            y=valid_data_except_holdout["scenes_count_in_train_and_valid"],
            groups=valid_data_except_holdout["location"],
        )
    ):
        valid_data_except_holdout.loc[valid_idx, "fold"] = fold_index

    print("Validation Only")
    for fold in valid_data_except_holdout.fold.unique():
        df = valid_data_except_holdout[valid_data_except_holdout.fold == fold]
        print("Fold", fold, len(df), set(df.location))
        print(
            df[
                [
                    "scene_id",
                    "platform_instances",
                    "vessel_instances",
                    "fishing_instances",
                    "unknown_vessel",
                    "unknown_fishing",
                    "location",
                    "average_platform_instances",
                    "average_vessel_instances",
                    "average_fishing_instances",
                ]
            ]
        )
        print()

    valid_data_with_holdout = pd.concat([valid_data_except_holdout, holdout])
    valid_data_with_holdout.to_csv("configs/dataset/valid_only_split.csv", index=False)

    df = XView3DataModule("data")
    for fold in range(4):
        train_df, valid_df, holdout_df, _ = df.train_val_split(
            splitter={"name": "precomputed", "split_csv": "configs/dataset/valid_only_split.csv"}, fold=fold, num_folds=4
        )

        train_df["near_shore"] = train_df["distance_from_shore_km"] <= 2
        valid_df["near_shore"] = valid_df["distance_from_shore_km"] <= 2

        print("Fold", fold)
        print("Confidence", "train", Counter(train_df.confidence), "valid", Counter(valid_df.confidence))
        print("Near shore", "train", Counter(train_df.near_shore), "valid", Counter(valid_df.near_shore))
        print("is_vessel ", "train", Counter(train_df.is_vessel), "valid", Counter(valid_df.is_vessel))
        print("is_fishing", "train", Counter(train_df.is_fishing), "valid", Counter(valid_df.is_fishing))

    holdout_df["near_shore"] = holdout_df["distance_from_shore_km"] <= 2
    print("Fold", "Holdout")
    print("Confidence", Counter(holdout_df.confidence))
    print("Near shore", Counter(holdout_df.near_shore))
    print("is_vessel ", Counter(holdout_df.is_vessel))
    print("is_fishing", Counter(holdout_df.is_fishing))

    # full_data_except_holdout = (
    #     scene_split[(scene_split.official_split != "test") & (scene_split.recommended_split.isin(["train", "validation"]))]
    #     .copy()
    #     .reset_index(drop=True)
    # )

    # skf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    # for fold_index, (_, valid_idx) in enumerate(
    #     skf.split(full_data_except_holdout, y=full_data_except_holdout.train_and_valid_instances, groups=full_data_except_holdout.location)
    # ):
    #     full_data_except_holdout.loc[valid_idx, "fold"] = fold_index

    # full_data_except_holdout.to_csv("full_data_except_holdout.csv", index=False)
    # print("Train + Validation")
    # for fold in full_data_except_holdout.fold.unique():
    #     df = full_data_except_holdout[full_data_except_holdout.fold == fold]
    #     print("Fold", fold, len(df), set(df.location))


if __name__ == "__main__":
    Fire(main)
