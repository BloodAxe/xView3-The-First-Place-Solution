import os
from functools import partial
from typing import List, Tuple, Optional, Callable, Dict, Any, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from pytorch_toolbelt.datasets import INPUT_IMAGE_KEY, INPUT_INDEX_KEY
from pytorch_toolbelt.utils.torch_utils import image_to_tensor
from sklearn.neighbors import BallTree
from sklearn.utils import compute_sample_weight
from torch.utils.data import Sampler, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from .bboxer import (
    MultilabelCircleNetCoder,
    MultilabelCircleNetEncodeResult,
)
from .constants import *
from ..constants import *

__all__ = [
    "MultilabelCircleNetDataModule",
    "MultilabelCircleNetFixedCropDataset",
    "MultilabelCircleNetRandomCropDataset",
    "centernet_collate",
]

from ..dataset import (
    RandomCropFromImageDataset,
    RandomCropAroundEachObjectDataset,
    XView3DataModule,
    KeypointsDataset,
    FixedCropFromImageDataset,
    read_multichannel_image,
)
from ..dataset.random_crop_dataset import encode_labels
from ..metric import get_shoreline_shoreline_contours


def centernet_collate(batch, channels_last=False):
    skip_keys = [
        TARGET_CENTERS_KEY,
        TARGET_FISHING_KEY,
        TARGET_VESSEL_KEY,
        TARGET_LABELS_KEY,
        TARGET_LENGTHS_KEY,
        INPUT_SCENE_ID_KEY,
        INPUT_SCENE_CROP_KEY,
        INPUT_CONFIDENCE_KEY,
        INPUT_LOCATION_ID_KEY,
        INPUT_IS_NEAR_SHORE_KEY,
    ]
    excluded_items = [dict((k, v) for k, v in b.items() if k in skip_keys) for b in batch]
    included_items = [dict((k, v) for k, v in b.items() if k not in skip_keys) for b in batch]

    batch: dict = default_collate(included_items)
    for k in skip_keys:
        out = [item[k] for item in excluded_items if k in item]
        if len(out):
            batch[k] = out

    # for key in [
    #     INPUT_IMAGE_KEY,
    #     CENTERNET_TARGET_OBJECTNESS_MAP,
    #     CENTERNET_OUTPUT_CLASS_MAP,
    #     CENTERNET_TARGET_OFFSET,
    #     CENTERNET_TARGET_SIZE,
    #     CENTERNET_TARGET_FISHING_MAP,
    #     CENTERNET_TARGET_VESSEL_MAP,
    # ]:
    #     if key in batch:
    #         batch[key] = batch[key].to(memory_format=torch.contiguous_format)

    if channels_last:
        batch[INPUT_IMAGE_KEY] = batch[INPUT_IMAGE_KEY].to(memory_format=torch.channels_last)
    return batch


class MultilabelCircleNetTargetsMixin:
    box_coder: MultilabelCircleNetCoder

    def build_centernet_sample(
        self,
        image,
        centers,
        confidences,
        lengths,
        is_vessel,
        is_fishing,
        index,
        image_id,
        is_near_shore,
        location,
        folder: str,
        crop_coords: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> Dict[str, Any]:
        # Mask NaN values
        image[~np.isfinite(image)] = 0

        result = {
            INPUT_INDEX_KEY: index,
            INPUT_IMAGE_KEY: image_to_tensor(image),
            TARGET_CENTERS_KEY: centers.tolist(),  # .tolist to prevent Catalyst converting it to Tensor and move to GPU
            TARGET_VESSEL_KEY: is_vessel.tolist(),
            TARGET_FISHING_KEY: is_fishing.tolist(),
            TARGET_LENGTHS_KEY: lengths.tolist(),
            INPUT_CONFIDENCE_KEY: confidences.tolist(),
            INPUT_IS_NEAR_SHORE_KEY: is_near_shore.tolist(),
            INPUT_LOCATION_ID_KEY: location,
            INPUT_FOLDER_KEY: folder,
            INPUT_SCENE_ID_KEY: image_id,
            TARGET_SAMPLE_WEIGHT: {"train": 0.25, "validation": 1.0, "valid": 1.0, "test": 0.1}[folder],
        }
        if crop_coords is not None:
            result[INPUT_SCENE_CROP_KEY] = crop_coords

        box_coder = self.box_coder.box_coder_for_image_size(image.shape)
        centernet_targets: MultilabelCircleNetEncodeResult = box_coder.encode(
            centers=centers, confidences=confidences, lengths=lengths, is_vessel=is_vessel, is_fishing=is_fishing
        )
        result[CENTERNET_TARGET_OBJECTNESS_MAP] = centernet_targets.heatmap
        result[CENTERNET_TARGET_VESSEL_MAP] = centernet_targets.is_vessel
        result[CENTERNET_TARGET_FISHING_MAP] = centernet_targets.is_fishing
        result[CENTERNET_TARGET_SIZE] = centernet_targets.lengths
        result[CENTERNET_TARGET_OFFSET] = centernet_targets.offset
        return result


class MultilabelCircleNetFixedCropDataset(FixedCropFromImageDataset, MultilabelCircleNetTargetsMixin):
    def __init__(
        self,
        scenes: List[str],
        centers: List[np.ndarray],
        locations: List[str],
        confidences: List[np.ndarray],
        is_vessel: List[np.ndarray],
        is_fishing: List[np.ndarray],
        is_near_shore: List[np.ndarray],
        lengths: List[np.ndarray],
        crop_coords: List[Tuple],
        scene_sizes: List[Tuple[int, int]],
        individual_transforms: Dict[str, A.Compose],
        normalization: Dict[str, A.ImageOnlyTransform],
        input_channels: List[str],
        transform: A.Compose,
        read_image_fn: Callable,
        box_coder: MultilabelCircleNetCoder,
        channels_last: bool = False,
    ):
        super().__init__(
            scenes=scenes,
            centers=centers,
            locations=locations,
            confidences=confidences,
            is_vessel=is_vessel,
            is_fishing=is_fishing,
            is_near_shore=is_near_shore,
            lengths=lengths,
            crop_coords=crop_coords,
            scene_sizes=scene_sizes,
            individual_transforms=individual_transforms,
            normalization=normalization,
            input_channels=input_channels,
            transform=transform,
            read_image_fn=read_image_fn,
            channels_last=channels_last,
        )
        self.box_coder = box_coder

    def build_sample(
        self,
        image,
        centers,
        confidences,
        is_vessel,
        is_fishing,
        lengths,
        index,
        folder: str,
        image_id,
        is_near_shore,
        location: str,
        crop_coords,
    ) -> Dict[str, Any]:
        return self.build_centernet_sample(
            image=image,
            centers=centers,
            confidences=confidences,
            is_vessel=is_vessel,
            is_fishing=is_fishing,
            lengths=lengths,
            index=index,
            image_id=image_id,
            is_near_shore=is_near_shore,
            location=location,
            folder=folder,
            crop_coords=crop_coords,
        )

    def get_collate_fn(self) -> Callable:
        return partial(centernet_collate, channels_last=self.channels_last)


class MultilabelCircleNetRandomCropDataset(RandomCropFromImageDataset, MultilabelCircleNetTargetsMixin):
    def __init__(
        self,
        scenes: List[str],
        shoreline_contours: List[np.ndarray],
        locations: List[str],
        centers: List[np.ndarray],
        confidences: List[np.ndarray],
        lengths: List[np.ndarray],
        is_vessel: List[np.ndarray],
        is_fishing: List[np.ndarray],
        is_near_shore: List[np.ndarray],
        scene_sizes: List[Tuple[int, int]],
        individual_transforms: Dict[str, A.Compose],
        normalization: Dict[str, A.ImageOnlyTransform],
        input_channels: List[str],
        transform: A.Compose,
        read_image_fn: Callable,
        crop_size: Tuple[int, int],
        box_coder: MultilabelCircleNetCoder,
        crop_around_ship_p=0.9,
        balance_near_shore: Union[None, bool, float] = None,
        balance_crowd=None,
        balance_by_type=False,
        channels_last=False,
    ):
        super().__init__(
            scenes=scenes,
            shoreline_contours=shoreline_contours,
            locations=locations,
            centers=centers,
            confidences=confidences,
            is_vessel=is_vessel,
            is_fishing=is_fishing,
            is_near_shore=is_near_shore,
            lengths=lengths,
            scene_sizes=scene_sizes,
            individual_transforms=individual_transforms,
            normalization=normalization,
            input_channels=input_channels,
            transform=transform,
            read_image_fn=read_image_fn,
            crop_size=crop_size,
            crop_around_ship_p=crop_around_ship_p,
            balance_near_shore=balance_near_shore,
            balance_crowd=balance_crowd,
            balance_by_type=balance_by_type,
            channels_last=channels_last,
        )
        self.box_coder = box_coder

    def build_sample(
        self,
        image,
        centers,
        confidences,
        is_vessel,
        is_fishing,
        lengths,
        index,
        folder: str,
        image_id,
        is_near_shore,
        location: str,
        crop_coords,
    ) -> Dict[str, Any]:
        return self.build_centernet_sample(
            image=image,
            centers=centers,
            confidences=confidences,
            is_vessel=is_vessel,
            is_fishing=is_fishing,
            lengths=lengths,
            index=index,
            folder=folder,
            image_id=image_id,
            is_near_shore=is_near_shore,
            location=location,
            crop_coords=crop_coords,
        )

    def get_collate_fn(self) -> Callable:
        return partial(centernet_collate, channels_last=self.channels_last)


class MultilabelCircleNetRandomCropAroundEachObjectDataset(RandomCropAroundEachObjectDataset, MultilabelCircleNetTargetsMixin):
    def __init__(
        self,
        scenes: List[str],
        locations: List[str],
        crop_centers: np.ndarray,
        centers: List[np.ndarray],
        confidences: List[np.ndarray],
        lengths: List[np.ndarray],
        is_vessel: List[np.ndarray],
        is_fishing: List[np.ndarray],
        is_near_shore: List[np.ndarray],
        scene_sizes: List[Tuple[int, int]],
        individual_transforms: Dict[str, A.Compose],
        normalization: Dict[str, A.ImageOnlyTransform],
        input_channels: List[str],
        transform: A.Compose,
        read_image_fn: Callable,
        crop_size: Tuple[int, int],
        box_coder: MultilabelCircleNetCoder,
        channels_last=False,
    ):
        super().__init__(
            scenes=scenes,
            locations=locations,
            crop_centers=crop_centers,
            centers=centers,
            confidences=confidences,
            is_vessel=is_vessel,
            is_fishing=is_fishing,
            is_near_shore=is_near_shore,
            lengths=lengths,
            scene_sizes=scene_sizes,
            individual_transforms=individual_transforms,
            normalization=normalization,
            input_channels=input_channels,
            transform=transform,
            read_image_fn=read_image_fn,
            crop_size=crop_size,
            channels_last=channels_last,
        )
        self.box_coder = box_coder

    def build_sample(
        self,
        image,
        centers,
        confidences,
        is_vessel,
        is_fishing,
        lengths,
        index,
        folder: str,
        image_id,
        is_near_shore,
        location: str,
        crop_coords,
    ) -> Dict[str, Any]:
        return self.build_centernet_sample(
            image=image,
            centers=centers,
            confidences=confidences,
            is_vessel=is_vessel,
            is_fishing=is_fishing,
            is_near_shore=is_near_shore,
            lengths=lengths,
            index=index,
            folder=folder,
            image_id=image_id,
            location=location,
            crop_coords=crop_coords,
        )

    def get_collate_fn(self) -> Callable:
        return partial(centernet_collate, channels_last=self.channels_last)


class MultilabelCircleNetDataModule(XView3DataModule):
    @classmethod
    def get_keypoint_params(cls):
        return A.KeypointParams(format="xys", label_fields=["confidences", "is_vessel", "is_fishing", "is_near_shore"])

    @classmethod
    def get_random_crop_each_object_dataset(
        cls,
        train_df: pd.DataFrame,
        train_image_size: Tuple[int, int],
        input_channels: List[str],
        box_coder: MultilabelCircleNetCoder,
        normalization: Dict[str, A.ImageOnlyTransform],
        num_samples: int = 1024,
        individual_augmentations: Optional[Dict] = None,
        augmentations: Optional[List[A.BasicTransform]] = None,
        balance_crowd=False,
        balance_per_scene=True,
        balance_near_shore=None,
        balance_vessel_type=False,
        balance_fishing_type=False,
        balance_by_type=False,
        balance_location=False,
        channels_last=False,
        read_image_fn=read_multichannel_image,
    ) -> Tuple[KeypointsDataset, Optional[Sampler]]:
        scene_sizes = dict(
            [
                (row["scene_id"], (row["rows"], row["cols"]))
                for i, row in pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "scene_sizes.csv")).iterrows()
            ]
        )

        all_images = []
        all_coordinates = []
        all_confidences = []
        all_lengths = []
        all_size = []
        all_is_vessel = []
        all_is_fishing = []
        all_is_near_shore = []

        all_scenes = []
        all_locations = []

        center_near_shore = []
        center_neighbors = []
        center_coordinates = []
        center_is_vessel = []
        center_is_fishing = []
        center_is_near_shore = []

        for scene_id in tqdm(train_df.scene_id.unique(), desc="Preparing training dataset"):
            # Get detections only for current scene
            scene_df = train_df[train_df.scene_id == scene_id]
            num_instances_in_scene = len(scene_df)
            scene_size = scene_sizes[scene_id]
            scene_path = scene_df.scene_path.values[0]
            location_id = scene_df.location.values[0]

            targets = cls.get_multilabel_targets_from_df(scene_df)
            distance_from_shore_km = scene_df.distance_from_shore_km
            near_shore = (distance_from_shore_km < 2).values

            tree = BallTree(targets.centers, leaf_size=8, metric="chebyshev")
            neighbors = tree.query_radius(targets.centers, r=max(train_image_size) // 2, count_only=True)

            all_scenes.extend([scene_id] * num_instances_in_scene)
            all_locations.extend([location_id] * num_instances_in_scene)
            all_images.extend([scene_path] * num_instances_in_scene)

            center_coordinates.extend(targets.centers)
            center_is_vessel.extend(targets.is_vessel)
            center_is_fishing.extend(targets.is_fishing)
            center_is_near_shore.extend(targets.is_near_shore)
            center_near_shore.extend(near_shore)
            center_neighbors.extend(neighbors)

            all_coordinates.extend([targets.centers] * num_instances_in_scene)
            all_confidences.extend([targets.confidences] * num_instances_in_scene)
            all_is_vessel.extend([targets.is_vessel] * num_instances_in_scene)
            all_is_fishing.extend([targets.is_fishing] * num_instances_in_scene)
            all_lengths.extend([targets.lengths] * num_instances_in_scene)
            all_is_near_shore.extend([targets.is_near_shore] * num_instances_in_scene)

            all_size.extend([scene_size] * num_instances_in_scene)

        if augmentations is None:
            augmentations = []

        transform = A.Compose(
            augmentations
            + [
                A.PadIfNeeded(
                    train_image_size[0],
                    train_image_size[1],
                    value=float("nan"),
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ],
            keypoint_params=cls.get_keypoint_params(),
        )
        train_ds = MultilabelCircleNetRandomCropAroundEachObjectDataset(
            scenes=all_images,
            locations=all_locations,
            centers=all_coordinates,
            confidences=all_confidences,
            crop_centers=np.array(center_coordinates),
            lengths=all_lengths,
            is_vessel=all_is_vessel,
            is_fishing=all_is_fishing,
            is_near_shore=all_is_near_shore,
            scene_sizes=all_size,
            individual_transforms=individual_augmentations,
            input_channels=input_channels,
            transform=transform,
            normalization=normalization,
            read_image_fn=read_image_fn,
            box_coder=box_coder,
            crop_size=train_image_size,
            channels_last=channels_last,
        )

        # weights = np.ones(len(all_coordinates), dtype=np.float64)
        # num_balancing = 0
        #
        # if balance_location:
        #     location_balance_weights = compute_sample_weight("balanced", all_locations)
        #     weights *= location_balance_weights
        #     num_balancing += 1
        #
        # if balance_crowd:
        #     inv_frequency = np.reciprocal(np.array(center_neighbors).astype(np.float32))
        #     crowd_balance_weights = np.sqrt(inv_frequency)  # Slightly dampen inverse proportional weight
        #     weights *= crowd_balance_weights
        #     num_balancing += 1
        #
        # if balance_per_scene:
        #     scene_balance_weights = compute_sample_weight("balanced", all_scenes)
        #     weights *= scene_balance_weights
        #     num_balancing += 1
        #
        # if balance_vessel_type:
        #     vessel_type_weights = compute_sample_weight({0: 0.45, 1: 0.45, IGNORE_LABEL: 0.1}, center_is_vessel)
        #     weights *= vessel_type_weights
        #     num_balancing += 1
        #
        # if balance_fishing_type:
        #     fishing_weights = compute_sample_weight({0: 0.45, 1: 0.45, IGNORE_LABEL: 0.1}, center_is_fishing)
        #     weights *= fishing_weights
        #     num_balancing += 1
        #
        # if balance_near_shore and len(np.bincount(center_near_shore)) > 1:
        #     shore_balance_weights = compute_sample_weight("balanced", center_near_shore)
        #     weights *= shore_balance_weights
        #     num_balancing += 1
        #
        # if num_balancing:
        #     weights **= 1.0 / num_balancing
        #

        weights = np.ones(len(all_coordinates), dtype=np.float64)
        num_balancing = 0

        if balance_location:
            location_balance_weights = compute_sample_weight("balanced", all_locations)
            weights += location_balance_weights
            num_balancing += 1

        if balance_crowd:
            inv_frequency = np.reciprocal(np.array(center_neighbors).astype(np.float32))
            crowd_balance_weights = inv_frequency
            if not np.all(np.isfinite(crowd_balance_weights)):
                raise RuntimeError()
            weights += crowd_balance_weights
            num_balancing += 1

        if balance_per_scene:
            scene_balance_weights = compute_sample_weight("balanced", all_scenes)
            weights += scene_balance_weights
            num_balancing += 1

        if balance_vessel_type:
            center_is_vessel = np.array(center_is_vessel)
            vessel_type_weights = compute_sample_weight(
                {
                    0: len(center_is_vessel) / (2.0 * np.sum(center_is_vessel == 0)),
                    1: len(center_is_vessel) / (2.0 * np.sum(center_is_vessel == 1)),
                    IGNORE_LABEL: 0,
                },
                center_is_vessel,
            )
            if not np.all(np.isfinite(vessel_type_weights)):
                raise RuntimeError()

            weights += vessel_type_weights
            num_balancing += 1

        if balance_fishing_type:
            center_is_fishing = np.array(center_is_fishing)
            fishing_weights = compute_sample_weight(
                {
                    0: len(center_is_fishing) / (2.0 * np.sum(center_is_fishing == 0)),
                    1: len(center_is_fishing) / (2.0 * np.sum(center_is_fishing == 1)),
                    IGNORE_LABEL: 0,
                },
                center_is_fishing,
            )
            if not np.all(np.isfinite(fishing_weights)):
                raise RuntimeError()
            weights += fishing_weights
            num_balancing += 1

        if balance_by_type:
            class_labels = encode_labels(center_is_vessel, center_is_fishing)
            type_balance_weights = compute_sample_weight("balanced", class_labels)
            weights += type_balance_weights
            num_balancing += 1

        if balance_near_shore and len(np.bincount(center_near_shore)) > 1:
            shore_balance_weights = compute_sample_weight("balanced", center_near_shore)
            weights += shore_balance_weights
            num_balancing += 1

        if num_balancing:
            weights /= float(num_balancing)

        train_sampler = WeightedRandomSampler(weights.astype(np.float32), num_samples=num_samples, replacement=True)
        return train_ds, train_sampler

    @classmethod
    def get_random_crop_training_dataset(
        cls,
        train_df: pd.DataFrame,
        train_image_size: Tuple[int, int],
        input_channels: List[str],
        box_coder: MultilabelCircleNetCoder,
        crop_around_ship_p: float,
        normalization: Dict[str, A.ImageOnlyTransform],
        shore_root: str,
        num_samples: int = 1024,
        individual_augmentations: Optional[Dict] = None,
        augmentations: Optional[List[A.BasicTransform]] = None,
        balance_near_shore=None,
        balance_crowd=None,
        balance_location=False,
        balance_by_type=False,
        balance_folder=False,
        channels_last=False,
        read_image_fn=read_multichannel_image,
    ) -> Tuple[KeypointsDataset, Optional[Sampler]]:
        scene_sizes = dict(
            [
                (row["scene_id"], (row["rows"], row["cols"]))
                for i, row in pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "scene_sizes.csv")).iterrows()
            ]
        )

        all_images = []
        all_centers = []
        all_confidences = []
        all_lengths = []
        all_size = []
        all_is_vessel = []
        all_is_fishing = []
        all_is_near_shore = []
        all_near_shore = []
        all_locations = []
        all_shoreline_contours = []
        all_folders = []

        for scene_id in tqdm(train_df.scene_id.unique(), desc="Preparing training dataset"):
            # Get detections only for current scene
            scene_df = train_df[train_df.scene_id == scene_id]
            scene_size = scene_sizes[scene_id]
            scene_path = scene_df.scene_path.values[0]
            location_id = scene_df.location.values[0]
            folder_id = scene_df.folder.values[0]

            shoreline_contours = get_shoreline_shoreline_contours(shore_root, scene_id)

            targets = cls.get_multilabel_targets_from_df(scene_df)
            distance_from_shore_km = scene_df.distance_from_shore_km
            near_shore = (distance_from_shore_km < 2).values

            all_images.append(scene_path)
            all_shoreline_contours.append(shoreline_contours)

            all_centers.append(targets.centers)
            all_confidences.append(targets.confidences)
            all_is_vessel.append(targets.is_vessel)
            all_is_fishing.append(targets.is_fishing)
            all_is_near_shore.append(targets.is_near_shore)
            all_lengths.append(targets.lengths)
            all_size.append(scene_size)
            all_near_shore.append(near_shore)

            all_locations.append(location_id)
            all_folders.append(folder_id)

        if augmentations is None:
            augmentations = []

        transform = A.Compose(
            augmentations
            + [
                A.PadIfNeeded(
                    train_image_size[0],
                    train_image_size[1],
                    value=float("nan"),
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ],
            keypoint_params=cls.get_keypoint_params(),
        )
        train_ds = MultilabelCircleNetRandomCropDataset(
            scenes=all_images,
            shoreline_contours=all_shoreline_contours,
            locations=all_locations,
            centers=all_centers,
            confidences=all_confidences,
            lengths=all_lengths,
            is_vessel=all_is_vessel,
            is_fishing=all_is_fishing,
            is_near_shore=all_near_shore,
            scene_sizes=all_size,
            individual_transforms=individual_augmentations,
            input_channels=input_channels,
            transform=transform,
            normalization=normalization,
            read_image_fn=read_image_fn,
            box_coder=box_coder,
            crop_size=train_image_size,
            crop_around_ship_p=crop_around_ship_p,
            balance_near_shore=balance_near_shore,
            balance_crowd=balance_crowd,
            balance_by_type=balance_by_type,
            channels_last=channels_last,
        )

        weights = len(train_df)
        num_balancing = 0

        if balance_location:
            location_balance_weights = compute_sample_weight("balanced", all_locations)
            weights *= location_balance_weights
            num_balancing += 1

        if balance_folder:
            folder_balance_weights = compute_sample_weight("balanced", all_folders)
            weights *= folder_balance_weights
            num_balancing += 1

        if num_balancing:
            weights **= 1.0 / num_balancing

        train_sampler = WeightedRandomSampler(np.ones(len(train_ds)), num_samples=num_samples, replacement=True)
        return train_ds, train_sampler

    @classmethod
    def get_validation_dataset(
        cls,
        valid_df: pd.DataFrame,
        valid_crop_size: Tuple[int, int],
        input_channels: List[str],
        box_coder: MultilabelCircleNetCoder,
        normalization: Dict[str, A.ImageOnlyTransform],
        channels_last=False,
        read_image_fn=read_multichannel_image,
    ) -> KeypointsDataset:
        """
        For validation dataset we apply sliding crops
        :param valid_df:
        :param kwargs:
        :return:
        """

        all_images = []
        all_centers = []
        all_confidences = []
        all_lengths = []
        all_crop_coords = []
        all_is_vessel = []
        all_is_fishing = []
        all_scene_sizes = []
        all_locations = []
        all_is_near_shore = []

        scene_sizes = dict(
            [
                (row["scene_id"], (row["rows"], row["cols"]))
                for i, row in pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "scene_sizes.csv")).iterrows()
            ]
        )

        for scene_id in tqdm(valid_df.scene_id.unique(), desc="Preparing validation dataset"):
            # Get detections only for current scene
            scene_df = valid_df[valid_df.scene_id == scene_id]
            scene_path = scene_df.scene_path.values[0]
            scene_size = scene_sizes[scene_id]
            location = scene_df.location.values[0]

            targets = cls.get_multilabel_targets_from_df(scene_df)
            for crop_coords in cls.iterate_crops(scene_size, valid_crop_size, valid_crop_size):
                all_images.append(scene_path)
                all_locations.append(location)
                all_centers.append(targets.centers)
                all_confidences.append(targets.confidences)
                all_is_vessel.append(targets.is_vessel)
                all_is_fishing.append(targets.is_fishing)
                all_is_near_shore.append(targets.is_near_shore)
                all_lengths.append(targets.lengths)
                all_crop_coords.append(crop_coords)
                all_scene_sizes.append(scene_size)

        individual_transforms = {}
        transform = A.Compose(
            [
                A.PadIfNeeded(
                    valid_crop_size[0],
                    valid_crop_size[1],
                    value=float("nan"),
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ],
            keypoint_params=cls.get_keypoint_params(),
        )
        valid_ds = MultilabelCircleNetFixedCropDataset(
            scenes=all_images,
            locations=all_locations,
            centers=all_centers,
            confidences=all_confidences,
            lengths=all_lengths,
            crop_coords=all_crop_coords,
            is_vessel=all_is_vessel,
            is_fishing=all_is_fishing,
            is_near_shore=all_is_near_shore,
            scene_sizes=all_scene_sizes,
            individual_transforms=individual_transforms,
            normalization=normalization,
            input_channels=input_channels,
            transform=transform,
            read_image_fn=read_image_fn,
            box_coder=box_coder,
            channels_last=channels_last,
        )
        return valid_ds
