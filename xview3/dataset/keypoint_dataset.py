import os
from collections import namedtuple
from typing import List, Callable, Dict, Tuple, Any

import albumentations as A
import numpy as np
from pytorch_toolbelt.utils import fs
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

__all__ = ["KeypointsDataset", "LoadImageResult"]

LoadImageResult = namedtuple(
    "LoadImageResult",
    ("individual_channels", "centers", "confidences", "is_vessel", "is_fishing", "lengths", "is_near_shore"),
)


class KeypointsDataset(Dataset):
    scenes: List[str]
    scene_ids: List[str]
    locations: List[str]

    centers: List[np.ndarray]
    confidences: List[np.ndarray]
    is_vessel: List[np.ndarray]
    is_fishing: List[np.ndarray]
    is_near_shore: List[np.ndarray]

    lengths: List[np.ndarray]
    scene_sizes: List[Tuple[int, int]]

    individual_transforms: Dict[str, A.Compose]
    input_channels: List[str]
    transform: A.Compose

    read_image_fn: Callable
    channels_last: bool

    def __init__(
        self,
        scenes: List[str],
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
        channels_last=False,
    ):
        if (
            len(scenes) != len(centers)
            or len(scenes) != len(locations)
            or len(scenes) != len(confidences)
            or len(scenes) != len(lengths)
            or len(scenes) != len(is_vessel)
            or len(scenes) != len(is_fishing)
            or len(scenes) != len(is_near_shore)
            or len(scenes) != len(scene_sizes)
        ):
            raise ValueError(
                f"Number of images ({len(scenes)}) must be equal number of keypoints ({len(centers)}) and number of labels ({len(is_vessel)})"
            )
        self.channels_last = channels_last
        self.scenes = scenes
        self.locations = locations
        self.scene_ids = list(map(fs.id_from_fname, scenes))
        self.folders = list(fs.id_from_fname(os.path.dirname(x)) for x in scenes)
        self.centers = centers
        self.confidences = confidences
        self.lengths = lengths
        self.scene_sizes = scene_sizes
        self.is_vessel = is_vessel
        self.is_fishing = is_fishing
        self.is_near_shore = is_near_shore
        self.transform = transform
        self.read_image_fn = read_image_fn

        self.individual_transforms = individual_transforms
        self.normalization = normalization
        self.input_channels = input_channels

    def load_image(self, index: int, crop_coords=None) -> LoadImageResult:
        individual_channels = self.read_image_fn(self.scenes[index], channels=self.input_channels, crop_coords=crop_coords)

        centers = self.centers[index]
        confidences = self.confidences[index]
        is_vessel = self.is_vessel[index]
        is_fishing = self.is_fishing[index]
        lengths = self.lengths[index]
        is_near_shore = self.is_near_shore[index]

        # Remove invisible keypoints & shift visible ones to crop origin
        if crop_coords is not None:
            (y1, y2), (x1, x2) = crop_coords
            mask = (y1 <= centers[:, 1]) & (centers[:, 1] < y2) & (x1 <= centers[:, 0]) & (centers[:, 0] < x2)
            centers = centers[mask] - np.array([x1, y1]).reshape((1, 2))
            confidences = confidences[mask]
            is_fishing = is_fishing[mask]
            is_near_shore = is_near_shore[mask]
            is_vessel = is_vessel[mask]
            lengths = lengths[mask]

        return LoadImageResult(
            individual_channels=individual_channels,
            centers=centers,
            confidences=confidences,
            is_vessel=is_vessel,
            is_fishing=is_fishing,
            lengths=lengths,
            is_near_shore=is_near_shore,
        )

    def apply_transformations(self, data: LoadImageResult) -> Dict[str, Any]:
        # Apply channel-specific augmentations
        channels = []
        for channel_name in self.input_channels:
            img = data.individual_channels[channel_name]
            if channel_name in self.individual_transforms:
                img = self.individual_transforms[channel_name](image=img)["image"]

            img = self.normalization[channel_name](image=img)["image"]
            channels.append(img)

        # Apply common augmentations
        image = np.dstack(channels)

        keypoints = [(cx, cy, length) for (cx, cy), length in zip(data.centers, data.lengths)]  # Keypoint format is x,y,s

        if (
            len(keypoints) != len(data.confidences)
            or len(keypoints) != len(data.is_vessel)
            or len(keypoints) != len(data.is_fishing)
            or len(keypoints) != len(data.is_near_shore)
            or len(keypoints) != len(data.confidences)
        ):
            raise RuntimeError("Detected mismatch in input data.Length of keypoints array not equal to present labels")

        data = self.transform(
            image=image,
            keypoints=keypoints,
            confidences=data.confidences,
            is_vessel=data.is_vessel,
            is_fishing=data.is_fishing,
            is_near_shore=data.is_near_shore,
        )
        return data

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError()

    def get_collate_fn(self) -> Callable:
        return default_collate

    def build_sample(
        self,
        image,
        centers,
        confidences,
        is_vessel,
        is_fishing,
        is_near_shore,
        lengths,
        index,
        folder: str,
        image_id: str,
        location: str,
        crop_coords: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> Dict[str, Any]:
        raise NotImplementedError()
