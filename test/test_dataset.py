import collections
import random
from typing import List

import albumentations as A
import glob
from collections import Counter, defaultdict

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import rasterio
import tifffile
from omegaconf import OmegaConf
from pytorch_toolbelt.utils import fs
from rasterio.windows import Window
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from xview3 import (
    build_augmentations,
    read_tiff_with_scaling,
    capped_randnorm,
    ignore_low_confidence_objects,
    filter_low_confidence_objects,
    MultilabelCircleNetDataModule,
    build_normalization,
    MultilabelCircleNetCoder,
)
from xview3.centernet.visualization import get_flatten_object_colors
from xview3.constants import (
    NODATA_VH_DB,
    IGNORE_LABEL,
    TARGET_FISHING_KEY,
    TARGET_VESSEL_KEY,
    INPUT_SCENE_ID_KEY,
    INPUT_LOCATION_ID_KEY,
    INPUT_IS_NEAR_SHORE_KEY,
)
from xview3.dataset import XView3DataModule, read_tiff
from xview3.dataset.splitters import FullDatasetWithHoldoutSplitter
from xview3.metric import compute_loc_performance, calculate_p_r_f


def test_dataset_read():
    owiMask = read_tiff("../data/tiny/train/cbe4ad26fe73f118t/owiMask.tif")
    bathymetry = read_tiff("../data/tiny/train/cbe4ad26fe73f118t/bathymetry.tif")
    print(np.unique(owiMask))
    print(np.min(bathymetry), np.max(bathymetry))


def test_splitter():
    splitter_cls = FullDatasetWithHoldoutSplitter("data")
    for f in range(4):
        train_df, valid_df, holdout_df, shroe_root = splitter_cls.train_test_split(fold=f, num_folds=4)

        is_vessel = train_df.is_vessel.values.astype(np.float32)
        is_vessel = is_vessel[np.isfinite(is_vessel)].astype(int)

        is_fishing = train_df.is_fishing.values.astype(np.float32)
        is_fishing = is_fishing[np.isfinite(is_fishing)].astype(int)

        print("Fold", f)
        print("Train")
        print("Scenes", len(train_df.scene_id.unique()), len(train_df.location.unique()))
        print("\tis_vessel ", np.bincount(is_vessel))
        print("\tis_fishing", np.bincount(is_fishing))

        is_vessel = valid_df.is_vessel.values.astype(np.float32)
        is_vessel = is_vessel[np.isfinite(is_vessel)].astype(int)

        is_fishing = valid_df.is_fishing.values.astype(np.float32)
        is_fishing = is_fishing[np.isfinite(is_fishing)].astype(int)

        print("Valid")
        print("Scenes", len(valid_df.scene_id.unique()), len(valid_df.location.unique()))
        print("\tis_vessel ", np.bincount(is_vessel))
        print("\tis_fishing", np.bincount(is_fishing))


def test_capped_randnorm():
    x = [capped_randnorm(-128, 128) for _ in range(8192)]
    plt.figure()
    plt.hist(x, bins=64)
    plt.show()


def compute_loc_score(pred, gt, method=compute_loc_performance):
    # For each scene, obtain the tp, fp, and fn indices for maritime
    # object detection in the *global* pred and gt dataframes
    tp_inds, fp_inds, fn_inds = [], [], []
    for scene_id in gt["scene_id"].unique():
        pred_sc = pred[pred["scene_id"] == scene_id]
        gt_sc = gt[gt["scene_id"] == scene_id]
        (
            tp_inds_sc,
            fp_inds_sc,
            fn_inds_sc,
        ) = method(pred_sc, gt_sc, distance_tolerance=200)

        tp_inds += tp_inds_sc
        fp_inds += fp_inds_sc
        fn_inds += fn_inds_sc
    loc_precision, loc_recall, loc_fscore = calculate_p_r_f(tp_inds, fp_inds, fn_inds)
    return loc_fscore


def test_vessel_length_distribution():
    df = pd.concat([pd.read_csv("../data/train.csv"), pd.read_csv("../data/validation.csv")])

    counts = []
    for scene in df.scene_id.unique():
        counts.append(len(df[df.scene_id == scene]))

    print(np.min(counts), np.mean(counts), np.std(counts), np.max(counts))
    vessel_length_m = df.vessel_length_m.values
    labels = XView3DataModule.encode_labels(df.is_vessel, df.is_fishing)
    print(np.bincount(labels))
    print(np.unique(labels))

    vessel_length_m = vessel_length_m[vessel_length_m > 0]

    log_len = np.log(vessel_length_m)
    print(np.min(vessel_length_m), np.min(log_len))
    print(np.max(vessel_length_m), np.max(log_len))

    plt.figure()
    plt.hist(vessel_length_m, bins=256)
    plt.yscale("log")
    plt.show()

    plt.figure()
    plt.hist(log_len, bins=256)
    # plt.yscale("log")
    plt.show()


@pytest.mark.parametrize("csv_fname", ["../data/train.csv", "../data/validation.csv"])
def test_target_balancing(csv_fname):
    df = pd.read_csv(csv_fname)
    scenes = df.scene_id.unique()

    df = df.drop(
        columns=[
            "left",
            "top",
            "right",
            "bottom",
            "detect_lat",
            "detect_lon",
            "detect_scene_row",
            "detect_scene_column",
            "detect_id",
        ]
    )

    # for scene in scenes:
    #     scene_df = df[df.scene_id == scene]
    #
    #     is_vessel = scene_df.is_vessel.values.astype(np.float32)
    #     is_fishing = scene_df.is_fishing.values.astype(np.float32)
    #     near_shore = scene_df.distance_from_shore_km.values < 2
    #
    #     print(
    #         "scene_id",
    #         scene,
    #         fs.id_from_fname(csv_fname),
    #         sum(is_vessel == 0),
    #         sum(is_vessel == 1),
    #         sum(is_fishing == 0),
    #         sum(is_fishing == 1),
    #         sum(near_shore == 0),
    #         sum(near_shore == 1),
    #         sum(near_shore & (is_vessel == 1)),
    #     )

    is_vessel = df.is_vessel.values.astype(np.float32)
    is_vessel = is_vessel[np.isfinite(is_vessel)].astype(int)

    is_fishing = df.is_fishing.values.astype(np.float32)
    is_fishing = is_fishing[np.isfinite(is_fishing)].astype(int)

    print()
    print(fs.id_from_fname(csv_fname))
    print("is_vessel ", np.bincount(is_vessel))
    print("is_fishing", np.bincount(is_fishing))
    # print("is_vessel == 0",sum(near_shore == 0))
    # print("is_vessel == 0",sum(near_shore == 1))
    # print("is_vessel == 0",sum(near_shore & (is_vessel == 1)))
    # print("is_vessel == 0",sum(near_shore & (is_fishing == 1)))


def test_target_encoding():
    train = pd.read_csv("../data/train.csv")
    train["split"] = "train"
    validation = pd.read_csv("../data/validation.csv")
    validation["split"] = "validation"

    df = pd.concat([train, validation])

    print()
    print("Train     ", np.bincount(LabelEncoder().fit_transform(train.confidence)))
    print("Validation", np.bincount(LabelEncoder().fit_transform(validation.confidence)))
    print()
    print("Is Vessel & Fishing", sum((df.is_vessel == 1) & (df.is_fishing == 1)))
    print("Is Vessel & Not Fishing", sum((df.is_vessel == 1) & (df.is_fishing == 0)))
    print("Is Vessel & Has Length", sum((df.is_vessel == 1) & (df.vessel_length_m >= 0)))
    print("Is Vessel & Nan Length", sum((df.is_vessel == 1) & (np.isnan(df.vessel_length_m))))

    print("Not Vessel & Fishing", sum((df.is_vessel == 0) & (df.is_fishing == 1)))
    print("Not Vessel & Not Fishing", sum((df.is_vessel == 0) & (df.is_fishing == 0)))
    print("Not Vessel & Has Length", sum((df.is_vessel == 0) & (df.vessel_length_m >= 0)))
    print("Not Vessel & Nan Length", sum((df.is_vessel == 0) & (np.isnan(df.vessel_length_m))))

    labels = XView3DataModule.encode_labels(df.is_vessel, df.is_fishing)
    lengths = XView3DataModule.encode_lengths(df.vessel_length_m)

    print("is_vessel == 0   ", sum(df.is_vessel == 0))
    print("is_vessel == 1   ", sum(df.is_vessel == 1))
    print("is_vessel == nan ", sum((df.is_vessel != 0) & (df.is_vessel != 1)))
    print("is_fishing == 0  ", sum(df.is_fishing == 0))
    print("is_fishing == 1  ", sum(df.is_fishing == 1))
    print("is_fishing == nan", sum((df.is_fishing != 0) & (df.is_fishing != 1)))

    plt.figure()
    plt.hist(df.loc[(df.is_vessel == 0) & (df.vessel_length_m > 0), "vessel_length_m"], bins=256)
    plt.title("Platform length")
    plt.yscale("log")
    plt.show()

    plt.figure()
    plt.hist(df.loc[(df.is_vessel == 1) & (df.vessel_length_m > 0), "vessel_length_m"], bins=256)
    plt.title("Vessel length")
    plt.yscale("log")
    plt.show()

    # print(
    #     "is both       nan",
    #     sum((df.is_vessel not in {True, False}) & (df.is_fishing not in {True, False})),
    # )

    dec_vessel_length_m = XView3DataModule.decode_lengths(lengths)

    assert np.all(labels >= 0)
    assert np.all(labels < 3)

    is_vessel, is_fishing = XView3DataModule.decode_labels(labels)

    is_vessel_mask = (df.is_vessel.values == 0) | (df.is_vessel.values == 1)
    np.testing.assert_array_equal(is_vessel[is_vessel_mask], df.is_vessel[is_vessel_mask].values)

    is_fishing_mask = ((df.is_fishing == 0) | (df.is_fishing == 1)).values
    np.testing.assert_array_equal(is_fishing[is_fishing_mask], df.is_fishing.values[is_fishing_mask])

    np.testing.assert_array_almost_equal(dec_vessel_length_m, df.vessel_length_m.fillna(0), decimal=3)

    print("Number of non-vessels", len(df[(df.is_vessel != 1)]))
    print(
        "Number of non-vessel with specified length",
        len(df[(df.is_vessel != 1) & (df.vessel_length_m > 0)]),
    )
    print("Number of vessels", len(df[(df.is_vessel == 1)]))
    print(
        "Number of vessel with specified length",
        len(df[(df.is_vessel == 1) & (df.vessel_length_m > 0)]),
    )


def test_min_max():
    depth = read_tiff("../data/tiny/train/72dba3e82f782f67t/bathymetry.tif")
    window = Window.from_slices((1000 // 50, 1512 // 50), (2000 // 50, 2512 / 50))
    with rasterio.open("../data/tiny/train/72dba3e82f782f67t/bathymetry.tif") as dataset:
        from rasterio.enums import Resampling

        frame = dataset.read(
            1,
            window=window,
            resampling=Resampling.bilinear,
            out_shape=(dataset.count, int(dataset.height * 50), int(dataset.width * 50)),
        )
    print(frame.shape)
    df = pd.read_csv("../s")


def test_dataset():
    dataset = XView3DataModule("D:/Develop/Kaggle/xView3/data", "tiny")
    train_df, valid_df, holdout, _ = dataset.train_val_split()
    print(len(train_df), len(valid_df))

    num_folds = 4
    for i in range(num_folds):
        train_df, valid_df, holdout, _ = dataset.train_val_split(i, num_folds)
        print(len(train_df), len(valid_df))

        is_vessel = valid_df.is_vessel.values.astype(np.float32)
        is_fishing = valid_df.is_fishing.values.astype(np.float32)
        near_shore = valid_df.distance_from_shore_km.values < 2

        print(
            "Fold",
            i,
            len(valid_df),
            sum(is_vessel == 0),
            sum(is_vessel == 1),
            sum(is_fishing == 0),
            sum(is_fishing == 1),
            sum(near_shore == 0),
            sum(near_shore == 1),
            sum(near_shore & (is_vessel == 1)),
            sum(near_shore & (is_fishing == 1)),
        )

    is_vessel = holdout.is_vessel.values.astype(np.float32)
    is_fishing = holdout.is_fishing.values.astype(np.float32)
    near_shore = holdout.distance_from_shore_km.values < 2
    print(
        "Holdout",
        sum(is_vessel == 0),
        sum(is_vessel == 1),
        sum(is_fishing == 0),
        sum(is_fishing == 1),
        sum(near_shore == 0),
        sum(near_shore == 1),
        sum(near_shore & (is_vessel == 1)),
        sum(near_shore & (is_fishing == 1)),
    )


def test_fatten_obj_colors():
    print(get_flatten_object_colors())


def test_tiff_io():
    ch1_fname = "../data/tiny/train/05bc615a9b0e1159t/VH_dB.tif"
    ch2_fname = "../data/tiny/train/05bc615a9b0e1159t/VV_dB.tif"

    image = np.dstack([read_tiff(ch1_fname), read_tiff(ch2_fname)])
    mask = image == NODATA_VH_DB
    min_val = image[~mask].min()
    max_val = image[~mask].max()
    # print(min_val,max_val)

    image = (255 * (image - min_val) / (max_val - min_val)).astype(np.uint8)
    image[mask] = 0
    image = np.dstack((image, np.zeros(image.shape[:2], dtype=np.uint8)))
    tifffile.imwrite("05bc615a9b0e1159t_rgb.tiff", image)

    crops = XView3DataModule.iterate_crops(image.shape[:2], tile_size=(2048, 2048), tile_step=(2048, 2048))
    for crop_index, crop in enumerate(crops):
        ch1_crop = read_tiff(ch1_fname, crop)
        ch2_crop = read_tiff(ch2_fname, crop)

        image_crop = np.dstack((ch1_crop, ch2_crop))
        mask_crop = image_crop == NODATA_VH_DB
        image_crop = (255 * (image_crop - min_val) / (max_val - min_val)).astype(np.uint8)
        image_crop[mask_crop] = 0
        image_crop = np.dstack((image_crop, np.zeros(image_crop.shape[:2], dtype=np.uint8)))

        tifffile.imwrite(f"05bc615a9b0e1159t_{crop_index}.tiff", image_crop)


def test_depth_tiff_io():
    # start_row = 225 * 50
    # start_col = 430 * 50
    # width = 4096
    # height = 4096
    # scene_id = "05bc615a9b0e1159t"

    start_row = 320 * 50
    start_col = 460 * 50
    width = 4096
    height = 4096
    scene_id = "e98ca5aba8849b06t"

    # start_row = 225 * 50
    # start_col = 430 * 50
    # width = 4096
    # height = 4096
    # scene_id = "72dba3e82f782f67t"

    # start_row = 100 * 50
    # start_col = 250 * 50
    # width = 4096
    # height = 4096
    # scene_id = "72dba3e82f782f67t"
    crop_coords = ((start_row, start_row + height), (start_col, start_col + width))

    bathymetry_full = read_tiff(f"../data/tiny/train/{scene_id}/bathymetry.tif")
    bathymetry_full[bathymetry_full == NODATA_VH_DB] = float("nan")

    mask_full = read_tiff(f"../data/tiny/train/{scene_id}/owiMask.tif")
    mask_full[mask_full == NODATA_VH_DB] = float("nan")
    mask_full = cv2.resize(
        mask_full,
        dsize=(bathymetry_full.shape[1], bathymetry_full.shape[0]),
        interpolation=cv2.INTER_LANCZOS4,
    )

    vv_full = read_tiff(f"../data/tiny/train/{scene_id}/VV_dB.tif")
    vv_full_shape = vv_full.shape[:2]
    vv_full[vv_full == NODATA_VH_DB] = float("nan")
    vv_full = cv2.resize(
        vv_full,
        dsize=(bathymetry_full.shape[1], bathymetry_full.shape[0]),
        interpolation=cv2.INTER_LANCZOS4,
    )

    vh_full = read_tiff(f"../data/tiny/train/{scene_id}/VH_dB.tif")
    vh_full[vh_full == NODATA_VH_DB] = float("nan")
    vh_full = cv2.resize(
        vh_full,
        dsize=(bathymetry_full.shape[1], bathymetry_full.shape[0]),
        interpolation=cv2.INTER_LANCZOS4,
    )

    pt1 = start_col // 50, start_row // 50
    pt2 = (start_col + width) // 50, (start_row + height) // 50

    cv2.rectangle(vv_full, pt1, pt2, float("nan"))
    cv2.rectangle(vh_full, pt1, pt2, float("nan"))
    cv2.rectangle(bathymetry_full, pt1, pt2, float("nan"))

    vv = read_tiff(f"../data/tiny/train/{scene_id}/VV_dB.tif", crop_coords=crop_coords)
    vh = read_tiff(f"../data/tiny/train/{scene_id}/VH_dB.tif", crop_coords=crop_coords)
    sy = vv_full_shape[0] / float(bathymetry_full.shape[0])
    sx = vv_full_shape[1] / float(bathymetry_full.shape[1])
    print(sx, sy)

    bathymetry = read_tiff_with_scaling(
        f"../data/tiny/train/{scene_id}/bathymetry.tif",
        crop_coords=crop_coords,
        s=(1.0 / sx, 1.0 / sy),
    )

    mask = read_tiff_with_scaling(
        f"../data/tiny/train/{scene_id}/owiMask.tif",
        crop_coords=crop_coords,
        s=(1.0 / sx, 1.0 / sy),
    )

    vv[vv == NODATA_VH_DB] = float("nan")
    vh[vh == NODATA_VH_DB] = float("nan")
    bathymetry[bathymetry == NODATA_VH_DB] = float("nan")
    mask[mask == NODATA_VH_DB] = float("nan")

    bathymetry = np.cbrt(bathymetry)
    f, ax = plt.subplots(2, 4, figsize=(18, 12))

    ax[0, 0].imshow(vv_full)
    ax[0, 0].axis("on")
    ax[0, 1].imshow(vh_full)
    ax[0, 1].axis("on")
    ax[0, 2].imshow(bathymetry_full)
    ax[0, 2].axis("on")
    ax[0, 3].imshow(mask_full)
    ax[0, 3].axis("on")

    ax[1, 0].imshow(vv)
    ax[1, 0].axis("off")
    ax[1, 1].imshow(vh)
    ax[1, 1].axis("off")
    ax[1, 2].imshow(bathymetry)
    ax[1, 2].axis("off")
    ax[1, 3].imshow(mask)
    ax[1, 3].axis("off")
    f.tight_layout()
    f.show()


@pytest.mark.parametrize("config_filename", glob.glob("../configs/augs/*.yaml"))
def test_augmentation_pipelines(config_filename):
    config = OmegaConf.load(config_filename)

    augmentations = build_augmentations(config.spatial)
    individual_augmentations = dict(
        [(key, A.Compose(build_augmentations(value))) for key, value in config.individual.items() if value is not None]
    )
    print(augmentations)
    print(individual_augmentations)


def dummy_read_multichannel_image(scene_id, channels: List[str], crop_coords=None):
    individual_channels = {}
    for channel in channels:
        (row_start, row_stop), (col_start, col_stop) = crop_coords
        shape = (row_stop - row_start), (col_stop - col_start)
        individual_channels[channel] = np.zeros(shape, dtype=np.float32)

    # individual_channels["diff(vv,vh)"] = individual_channels["vv"] - individual_channels["vh"]
    # individual_channels["mean(vv,vh)"] = 0.5 * (individual_channels["vv"] + individual_channels["vh"])
    return individual_channels


@pytest.mark.parametrize(
    (
        "balance_near_shore",
        "balance_crowd",
        "balance_location",
        "balance_per_scene",
        "balance_vessel_type",
        "balance_fishing_type",
        "balance_by_type",
        "crop_around_ship_p",
        "sampler_type",
    ),
    [
        # (False, False, False, False, False, False, False, 0.8, "fixed_crop"),
        # (False, False, False, False, False, False, 0.8, "random_crop"),
        #
        # (False, False, False, False, False, False, True, 0.8, "random_crop"),
        (False, False, False, False, False, False, True, 0.8, "each_object"),
        # (False, True, False, False, False, False, 0.8, "random_crop"),
        # (False, False, True, False, False, False, 0.8, "random_crop"),
        # (False, False, False, True, False, False, 0.8, "random_crop"),
        # (False, False, False, False, True, False, 0.8, "random_crop"),
        # (False, False, False, False, False, True, 0.8, "random_crop"),
        #
        # (True, True, True, False, True, True, 0.8, "random_crop"),
        # #
        # (False, False, False, False, False, False, 0.8, "each_object"),
        # #
        # (True, False, False, False, False, False, 0.8, "each_object"),
        # (False, True, False, False, False, False, 0.8, "each_object"),
        # (False, False, True, False, False, False, 0.8, "each_object"),
        # (False, False, False, True, False, False, 0.8, "each_object"),
        # (False, False, False, False, True, False, 0.8, "each_object"),
        # (False, False, False, False, False, True, 0.8, "each_object"),
        # #
        # (True, True, True, False, True, True, 0.8, "each_object"),
    ],
)
def test_dataset_balancing(
    balance_near_shore: bool,
    balance_crowd: bool,
    balance_location: bool,
    balance_per_scene: bool,
    balance_vessel_type: bool,
    balance_fishing_type: bool,
    balance_by_type: bool,
    crop_around_ship_p: float,
    sampler_type: str,
    fold=0,
    num_folds=4,
    train_image_size=1024,
    num_samples=4096,
):
    random.seed(42)
    data_dir = "../data"

    dataset = MultilabelCircleNetDataModule(data_dir=data_dir)
    train_df, valid_df, _, shore_root = dataset.train_val_split(
        splitter={"name": "precomputed", "split_csv": "../configs/dataset/valid_only_split.csv"},
        fold=fold,
        num_folds=num_folds,
    )

    print(collections.Counter(valid_df.is_fishing))

    # if config.ignore_low_confidence_objects:
    #     train_df = ignore_low_confidence_objects(train_df)
    #     valid_df = ignore_low_confidence_objects(valid_df)

    augs_config = OmegaConf.load("../configs/augs/none.yaml")
    augmentations = build_augmentations(augs_config.spatial)
    normalization = build_normalization(OmegaConf.load("../configs/normalization/default.yaml"))

    box_coder = MultilabelCircleNetCoder(
        image_size=(train_image_size, train_image_size),
        output_stride=4,
        max_objects=512,
        heatmap_encoding="umich",
        labels_encoding="circle",
        ignore_value=IGNORE_LABEL,
        fixed_radius=3,
        labels_radius=1,
    )

    individual_augmentations = dict()

    if sampler_type == "each_object":
        train_ds, train_sampler = dataset.get_random_crop_each_object_dataset(
            train_df,
            train_image_size=(train_image_size, train_image_size),
            input_channels=["vv", "vh"],
            box_coder=box_coder,
            num_samples=num_samples,
            individual_augmentations=individual_augmentations,
            augmentations=augmentations,
            normalization=normalization,
            balance_crowd=balance_crowd,
            balance_near_shore=balance_near_shore,
            balance_vessel_type=balance_vessel_type,
            balance_fishing_type=balance_fishing_type,
            balance_by_type=balance_by_type,
            balance_per_scene=balance_per_scene,
            balance_location=balance_location,
            read_image_fn=dummy_read_multichannel_image,
        )
    elif sampler_type == "fixed_crop":
        train_ds = dataset.get_validation_dataset(
            train_df,
            valid_crop_size=(train_image_size, train_image_size),
            input_channels=["vv", "vh"],
            read_image_fn=dummy_read_multichannel_image,
            box_coder=box_coder,
            normalization=normalization,
        )
        train_sampler = None
    else:
        train_ds, train_sampler = dataset.get_random_crop_training_dataset(
            train_df,
            train_image_size=(train_image_size, train_image_size),
            input_channels=["vv", "vh"],
            box_coder=box_coder,
            num_samples=num_samples,
            crop_around_ship_p=crop_around_ship_p,
            individual_augmentations=dict(),
            augmentations=augmentations,
            normalization=normalization,
            balance_near_shore=balance_near_shore,
            balance_crowd=balance_crowd,
            balance_by_type=balance_by_type,
            balance_location=balance_location,
            read_image_fn=dummy_read_multichannel_image,
        )

    scene_counts = defaultdict(int)
    location_counts = defaultdict(int)
    fishing_counter = defaultdict(int)
    vessel_counter = defaultdict(int)
    near_shore_counter = defaultdict(int)

    for batch in DataLoader(
        train_ds, batch_size=16, num_workers=4, drop_last=False, sampler=train_sampler, collate_fn=train_ds.get_collate_fn()
    ):
        is_fishing_batch = batch[TARGET_FISHING_KEY]
        is_vessel_batch = batch[TARGET_VESSEL_KEY]
        scene_id_batch = batch[INPUT_SCENE_ID_KEY]
        location_id_batch = batch[INPUT_LOCATION_ID_KEY]
        is_near_shore_batch = batch[INPUT_IS_NEAR_SHORE_KEY]

        for location_id in location_id_batch:
            location_counts[location_id] += 1

        for scene_id in scene_id_batch:
            scene_counts[scene_id] += 1

        for is_vessel_vec in is_vessel_batch:
            for x in is_vessel_vec:
                vessel_counter[x] += 1

        for is_fishing_vec in is_fishing_batch:
            for x in is_fishing_vec:
                fishing_counter[x] += 1

        for is_near_shore_vec in is_near_shore_batch:
            for x in is_near_shore_vec:
                near_shore_counter[x] += 1

    print()
    print(
        "balance_near_shore",
        balance_near_shore,
        "balance_crowd",
        balance_crowd,
        "balance_location",
        balance_location,
        "balance_per_scene",
        balance_per_scene,
        "balance_vessel_type",
        balance_vessel_type,
        "balance_fishing_type",
        balance_fishing_type,
        "crop_around_ship_p",
        crop_around_ship_p,
        "sampler_type",
        sampler_type,
    )
    print(location_counts)
    print("vessel_counter ", "Platform   ", vessel_counter[0], "Vessel ", vessel_counter[1], "N/A", vessel_counter[IGNORE_LABEL])
    print("fishing_counter", "Non Fishing", fishing_counter[0], "Fishing", fishing_counter[1], "N/A", fishing_counter[IGNORE_LABEL])
    print("near_shore_counter", "Offshore", near_shore_counter[0], "Near Shore", near_shore_counter[1])
    print()
