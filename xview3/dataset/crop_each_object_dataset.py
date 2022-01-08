from typing import List, Callable, Dict, Tuple, Any

import albumentations as A
import numpy as np

from .keypoint_dataset import KeypointsDataset
from .random_crop_dataset import capped_randnorm

__all__ = ["RandomCropAroundEachObjectDataset"]


class RandomCropAroundEachObjectDataset(KeypointsDataset):
    crop_size: Tuple[int, int]

    def __init__(
        self,
        scenes: List[str],
        locations: List[str],
        crop_centers: np.ndarray,
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
        channels_last=False,
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
        self.crop_centers = crop_centers
        self.crop_size = crop_size

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Get ships of the current scene
        scene_rows, scene_cols = self.scene_sizes[index]
        crop_center = self.crop_centers[index]

        col, row = crop_center

        dx = int(capped_randnorm(-self.crop_size[0] // 4, self.crop_size[0] // 4))
        dy = int(capped_randnorm(-self.crop_size[1] // 4, self.crop_size[1] // 4))

        start_row = max(0, row + dx - self.crop_size[0] // 2)
        start_col = max(0, col + dy - self.crop_size[1] // 2)

        end_row = min(scene_rows, start_row + self.crop_size[0])
        end_col = min(scene_cols, start_col + self.crop_size[1])

        crop_coords = (start_row, end_row), (start_col, end_col)

        data = self.load_image(index, crop_coords=crop_coords)
        data = self.apply_transformations(data)

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
