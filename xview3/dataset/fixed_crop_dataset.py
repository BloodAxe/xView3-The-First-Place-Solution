from typing import List, Callable, Dict, Tuple, Any

import albumentations as A
import numpy as np

from .keypoint_dataset import KeypointsDataset

__all__ = ["FixedCropFromImageDataset"]


class FixedCropFromImageDataset(KeypointsDataset):
    crop_coords = None

    def __init__(
        self,
        scenes: List[str],
        locations: List[str],
        centers: List[np.ndarray],
        confidences: List[np.ndarray],
        is_vessel: List[np.ndarray],
        is_fishing: List[np.ndarray],
        is_near_shore: List[np.ndarray],
        crop_coords: List[Tuple],
        lengths: List[np.ndarray],
        scene_sizes: List[Tuple[int, int]],
        individual_transforms: Dict[str, A.Compose],
        normalization: Dict[str, A.ImageOnlyTransform],
        input_channels: List[str],
        transform: A.Compose,
        read_image_fn: Callable,
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
            input_channels=input_channels,
            normalization=normalization,
            transform=transform,
            read_image_fn=read_image_fn,
            channels_last=channels_last,
        )
        self.crop_coords = crop_coords

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        crop_coords = self.crop_coords[index]

        data = self.load_image(index, crop_coords=crop_coords)
        data = self.apply_transformations(data)

        image = data["image"]
        keypoints = np.asarray(data["keypoints"], dtype=np.float32).reshape((-1, 3))
        is_vessel = np.asarray(data["is_vessel"], dtype=np.long).reshape((-1))
        is_fishing = np.asarray(data["is_fishing"], dtype=np.long).reshape((-1))
        is_near_shore = np.asarray(data["is_near_shore"], dtype=np.long).reshape((-1))
        confidences = np.asarray(data["confidences"], dtype=np.object).reshape((-1))

        image_id = self.scene_ids[index]
        folder = self.folders[index]

        if len(keypoints) != len(is_vessel):
            raise RuntimeError("Number of is_vessel does not equal to number of labels")
        if len(keypoints) != len(is_fishing):
            raise RuntimeError("Number of is_fishing does not equal to number of labels")
        if len(keypoints) != len(is_near_shore):
            raise RuntimeError("Number of is_near_shore does not equal to number of labels")
        if len(keypoints) != len(is_near_shore):
            raise RuntimeError("Number of is_near_shore does not equal to number of labels")
        if len(keypoints) != len(confidences):
            raise RuntimeError("Number of confidences does not equal to number of labels")

        centers = keypoints[:, :2].reshape((-1, 2))
        lengths = keypoints[:, 2].reshape((-1))
        return self.build_sample(
            image=image,
            centers=centers,
            confidences=confidences,
            is_vessel=is_vessel,
            is_fishing=is_fishing,
            lengths=lengths,
            folder=folder,
            index=index,
            image_id=image_id,
            is_near_shore=is_near_shore,
            location=self.locations[index],
            crop_coords=crop_coords,
        )
