import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from fire import Fire
from pytorch_toolbelt.utils import fs
from tqdm import tqdm

from xview3.constants import NODATA_VV_DB
from xview3.dataset import read_tiff


def image_stats(image: np.ndarray):
    vv_mask = (image != NODATA_VV_DB).astype(np.uint8)
    vv_min, vv_max, _, _ = cv2.minMaxLoc(image, mask=vv_mask)
    vv_mean, vv_std = cv2.meanStdDev(image, mask=vv_mask)
    return float(vv_min), float(vv_max), float(vv_mean), float(vv_std)


def main(data_dir: str):
    scene_size = defaultdict(list)

    train_scenes = os.path.join(data_dir, "full", "train")
    valid_scenes = os.path.join(data_dir, "full", "validation")

    dirs = [os.path.join(train_scenes, x) for x in sorted(os.listdir(train_scenes)) if os.path.isdir(os.path.join(train_scenes, x))] + [
        os.path.join(valid_scenes, x) for x in sorted(os.listdir(valid_scenes)) if os.path.isdir(os.path.join(valid_scenes, x))
    ]
    print(len(dirs))

    for scene_id in tqdm(dirs):
        vh_path = os.path.join(scene_id, "VH_dB.tif")
        vv_path = os.path.join(scene_id, "VV_dB.tif")
        vh = read_tiff(vh_path)
        vv = read_tiff(vv_path)

        if vh.shape != vv.shape:
            print(vh_path, vh_path, "Size mismatch")
            print(vh.shape, vv.shape)

        vv_min, vv_max, vv_mean, vv_std = image_stats(vv)
        vh_min, vh_max, vh_mean, vh_std = image_stats(vh)

        scene_size["scene_id"].append(fs.id_from_fname(scene_id))
        scene_size["rows"].append(vh.shape[0])
        scene_size["cols"].append(vh.shape[1])
        scene_size["split"].append("train" if "train" in vh_path else "validation")

        scene_size["vv_min"].append(vv_min)
        scene_size["vv_max"].append(vv_max)
        scene_size["vv_mean"].append(vv_mean)
        scene_size["vv_std"].append(vv_std)

        scene_size["vh_min"].append(vh_min)
        scene_size["vh_max"].append(vh_max)
        scene_size["vh_mean"].append(vh_mean)
        scene_size["vh_std"].append(vh_std)

    scene_size = pd.DataFrame.from_dict(scene_size)
    scene_size.to_csv("scene_sizes.csv", index=False)


if __name__ == "__main__":
    Fire(main)
