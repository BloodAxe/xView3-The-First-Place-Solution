import os
from functools import partial

import cv2
import numpy as np
import pandas as pd
from fire import Fire
from tqdm import tqdm

from xview3 import XView3DataModule, read_multichannel_image, SigmoidNormalization
from xview3.centernet.visualization import (
    vis_detections_opencv,
    create_false_color_composite,
)
from xview3.constants import PIX_TO_M


def rasterize_test(test, prefix):
    normalize = SigmoidNormalization()
    test_scenes = list(os.listdir(test))
    for scene_id in tqdm(test_scenes):
        scene_path = os.path.join(test, scene_id)
        image = read_multichannel_image(scene_path, ["vv", "vh"])
        size_down_4 = image["vv"].shape[1] // 4, image["vv"].shape[0] // 4
        image_rgb = create_false_color_composite(
            normalize(image=cv2.resize(image["vv"], dsize=size_down_4, interpolation=cv2.INTER_LINEAR))["image"],
            normalize(image=cv2.resize(image["vh"], dsize=size_down_4, interpolation=cv2.INTER_LINEAR))["image"],
        )
        image_rgb[~np.isfinite(image_rgb)] = 0
        cv2.imwrite(prefix + scene_id + ".jpg", image_rgb)


def rasterize(dataset, prefix=""):
    normalize = SigmoidNormalization()
    for scene_id in tqdm(dataset.scene_id.unique()):
        # Get detections only for current scene
        scene_df = dataset[dataset.scene_id == scene_id]
        scene_path = scene_df.scene_path.values[0]

        image = read_multichannel_image(scene_path, ["vv", "vh"])

        size_down_4 = image["vv"].shape[1] // 4, image["vv"].shape[0] // 4
        image_rgb = create_false_color_composite(
            normalize(image=cv2.resize(image["vv"], dsize=size_down_4, interpolation=cv2.INTER_AREA))["image"],
            normalize(image=cv2.resize(image["vh"], dsize=size_down_4, interpolation=cv2.INTER_AREA))["image"],
        )
        image_rgb[~np.isfinite(image_rgb)] = 0

        centers, true_labels, lengths = XView3DataModule.get_label_targets_from_df(scene_df)

        centers = (centers * 0.25).astype(int)
        gt_is_vessel, gt_is_fishing = XView3DataModule.decode_labels(true_labels)

        image_rgb = vis_detections_opencv(
            image_rgb,
            centers=centers,
            lengths=XView3DataModule.decode_lengths(lengths) / PIX_TO_M,
            is_vessel_vec=gt_is_vessel,
            is_fishing_vec=gt_is_fishing,
            scores=np.ones(len(centers)),
            show_title=True,
            alpha=0.1,
        )
        cv2.imwrite(prefix + scene_id + ".jpg", image_rgb)


def main(data_dir: str = "f:/xview3"):
    def append_prefix(x, folder):
        return os.path.join(data_dir, "full", folder, x)

    os.makedirs("visualizations", exist_ok=True)

    # train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    # train_df["scene_path"] = train_df["scene_id"].apply(partial(append_prefix, folder="train"))
    # rasterize(train_df, prefix="visualizations/train_")

    # valid_df = pd.read_csv(os.path.join(data_dir, "validation.csv"))
    # valid_df["scene_path"] = valid_df["scene_id"].apply(partial(append_prefix, folder="validation"))
    # rasterize(valid_df, prefix="visualizations/validation_")

    rasterize_test("f:/xview3/test", prefix="visualizations/test_")


if __name__ == "__main__":
    Fire(main)
