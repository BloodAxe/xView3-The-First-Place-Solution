import random
from typing import List, Callable, Dict, Tuple, Any, Union, Optional

import albumentations as A
import numpy as np
import math

from sklearn.neighbors import BallTree
from sklearn.utils import compute_sample_weight

from .keypoint_dataset import KeypointsDataset, LoadImageResult

__all__ = ["RandomCropFromImageDataset", "capped_randnorm"]


def capped_randnorm(low, high):
    mean = random.normalvariate((low + high) * 0.5, sigma=math.sqrt(high - low))
    return np.clip(mean, low, high)


def encode_labels(is_vessel, is_fishing):
    v = np.zeros_like(is_vessel, dtype=int)
    v[is_vessel == 0] = 1
    v[is_vessel == 1] = 2
    v[is_fishing == 0] = 2
    v[is_fishing == 1] = 3

    return v


class RandomCropFromImageDataset(KeypointsDataset):
    crop_around_ship_p: float
    crop_size: Tuple[int, int]
    balance_near_shore: Union[None, bool, float]
    shoreline_contours: List[np.ndarray]

    def __init__(
        self,
        scenes: List[str],
        shoreline_contours: List[np.ndarray],
        locations: List[str],
        centers: List[np.ndarray],
        confidences: List[np.ndarray],
        is_vessel: List[np.ndarray],
        is_fishing: List[np.ndarray],
        is_near_shore: List[np.ndarray],
        lengths: List[np.ndarray],
        scene_sizes: List[Tuple[int, int]],
        individual_transforms: Dict[str, A.Compose],
        normalization: Dict[str, A.ImageOnlyTransform],
        input_channels: List[str],
        transform: A.Compose,
        read_image_fn: Callable,
        crop_size: Tuple[int, int],
        crop_around_ship_p: float = 0.9,
        balance_near_shore: Union[None, bool, float] = None,
        balance_crowd: Union[None, bool, float] = False,
        balance_by_type=False,
        channels_last: bool = False,
    ):
        super().__init__(
            scenes=scenes,
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
            channels_last=channels_last,
        )
        self.crop_size = crop_size
        self.crop_around_ship_p = crop_around_ship_p
        self.shoreline_contours = shoreline_contours

        all_weights = []
        for index in range(len(scenes)):
            weights = np.ones(len(centers[index]), dtype=np.float32)
            num_balancing = 0
            if balance_near_shore and len(np.bincount(is_near_shore[index])) > 1:
                shore_balance_weights = compute_sample_weight("balanced", is_near_shore[index])
                shore_balance_weights = shore_balance_weights / shore_balance_weights.sum()
                weights += shore_balance_weights
                num_balancing += 1

            if balance_crowd:
                tree = BallTree(centers[index], leaf_size=8, metric="chebyshev")
                nbhs = tree.query_radius(centers[index], r=max(crop_size) // 2, count_only=True)
                inv_frequency = np.reciprocal(nbhs.astype(np.float32))
                crowd_balance_weights = inv_frequency / inv_frequency.sum()
                weights += crowd_balance_weights
                num_balancing += 1

            if balance_by_type:
                class_labels = encode_labels(is_vessel[index], is_fishing[index])
                type_balance_weights = compute_sample_weight("balanced", class_labels)
                weights += type_balance_weights
                num_balancing += 1

            if num_balancing:
                weights /= num_balancing

            all_weights.append(weights)
        self.weights = all_weights

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Get ships of the current scene
        scene_rows, scene_cols = self.scene_sizes[index]
        shoreline_contours = self.shoreline_contours[index]  # [(row, column)] / [(y,x)]

        # Weights of the each sample in scene
        weights = self.weights[index]

        targets: Optional[LoadImageResult] = None
        should_crop = True
        while should_crop:
            centers = self.centers[index]
            if len(centers) and random.random() < self.crop_around_ship_p:
                if len(weights) != len(centers):
                    print("Size mismatch", index, self.scene_ids[index], len(weights), len(centers))
                random_ship_index = random.choices(range(len(centers)), weights)[0]
                col, row = centers[random_ship_index]

                dx = int(capped_randnorm(-self.crop_size[0] // 4, self.crop_size[0] // 4))
                dy = int(capped_randnorm(-self.crop_size[1] // 4, self.crop_size[1] // 4))

                start_row = max(0, row + dx - self.crop_size[0] // 2)
                start_col = max(0, col + dy - self.crop_size[1] // 2)

                end_row = min(scene_rows, start_row + self.crop_size[0])
                end_col = min(scene_cols, start_col + self.crop_size[1])

                crop_coords = (start_row, end_row), (start_col, end_col)
            else:
                # There is 50% chance to sample from near-shore location or from the entire scene
                if len(shoreline_contours) and random.random() < 0.5:
                    row, col = random.choice(shoreline_contours)
                    row = int(row)
                    col = int(col)

                    dx = int(capped_randnorm(-self.crop_size[0] // 4, self.crop_size[0] // 4))
                    dy = int(capped_randnorm(-self.crop_size[1] // 4, self.crop_size[1] // 4))

                    start_row = max(0, row + dx - self.crop_size[0] // 2)
                    start_col = max(0, col + dy - self.crop_size[1] // 2)

                    end_row = min(scene_rows, start_row + self.crop_size[0])
                    end_col = min(scene_cols, start_col + self.crop_size[1])

                    crop_coords = (start_row, end_row), (start_col, end_col)
                else:
                    start_row = random.randint(0, scene_rows - self.crop_size[0])
                    start_col = random.randint(0, scene_cols - self.crop_size[1])
                    crop_coords = (start_row, start_row + self.crop_size[0]), (
                        start_col,
                        start_col + self.crop_size[1],
                    )

            targets = self.load_image(index, crop_coords=crop_coords)

            # We repeat random cropping process to ensure at least 20% of the tile has signal
            # missing_pixels = (~np.isfinite(targets.individual_channels["vv"])) | (~np.isfinite(targets.individual_channels["vh"]))
            # Missing data is identical in vv&vh. So check only one channel.
            if "sar" in targets.individual_channels:
                missing_pixels = ~np.isfinite(targets.individual_channels["sar"])
            else:
                missing_pixels = ~np.isfinite(targets.individual_channels["vv"])

            missing_pixels_ratio = float(missing_pixels.sum()) / np.prod(missing_pixels.shape)
            should_crop = missing_pixels_ratio > 0.8

        data = self.apply_transformations(targets)

        image = data["image"]
        keypoints = np.asarray(data["keypoints"], dtype=np.float32).reshape((-1, 3))
        is_vessel = np.asarray(data["is_vessel"], dtype=np.long).reshape((-1))
        is_fishing = np.asarray(data["is_fishing"], dtype=np.long).reshape((-1))
        is_near_shore = np.asarray(data["is_near_shore"], dtype=np.long).reshape((-1))
        confidences = np.asarray(data["confidences"], dtype=np.object).reshape((-1))

        if len(keypoints) != len(is_vessel):
            raise RuntimeError("Number of is_vessel does not equal to number of keypoints")
        if len(keypoints) != len(is_fishing):
            raise RuntimeError("Number of is_fishing does not equal to number of keypoints")

        centers = keypoints[:, :2].reshape((-1, 2))
        lengths = keypoints[:, 2].reshape((-1))
        return self.build_sample(
            image=image,
            centers=centers,
            confidences=confidences,
            is_vessel=is_vessel,
            is_fishing=is_fishing,
            is_near_shore=is_near_shore,
            lengths=lengths,
            index=index,
            folder=self.folders[index],
            image_id=self.scene_ids[index],
            location=self.locations[index],
            crop_coords=crop_coords,
        )
