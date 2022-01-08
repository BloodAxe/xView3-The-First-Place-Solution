import math
import random
from typing import List, Union, Tuple, Callable

import albumentations as A
import cv2
import numpy as np
from sklearn.utils import compute_sample_weight

__all__ = ["CopyPasteAugmentation"]


class CopyPasteAugmentation(A.DualTransform):
    def __init__(
        self,
        images: List[str],
        bboxes: List,
        labels: List[int],
        transform: A.Compose,
        read_image_fn: Callable,
        class_weights: Union[str, np.ndarray] = "balanced",
        seamless_clone_p=0,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.seamless_clone_p = seamless_clone_p
        self.images = images
        self.bboxes = np.asarray(bboxes, dtype=int)
        self.labels = np.asarray(labels, dtype=int)
        self.transform = transform
        self.read_image = read_image_fn
        if class_weights == "balanced":
            sample_weights = compute_sample_weight("balanced", labels)
        else:
            sample_weights = compute_sample_weight(class_weights, labels)

        self.sample_weights = sample_weights

    @property
    def targets(self):
        return {
            "image": self.apply,
            "bboxes": self.apply_to_bboxes,
        }

    @property
    def targets_as_params(self):
        return "image", "bboxes"

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        rows, cols = image.shape[:2]
        bboxes = params["bboxes"]
        bboxes = A.convert_bboxes_from_albumentations(bboxes, "pascal_voc", rows, cols)

        # Compute average object size
        if len(bboxes) != 0:
            bboxes = np.array(bboxes)[:, :4]
            median_size = np.median(np.sqrt((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])))
        else:
            median_size = None

        seed_image, seed_label = self._select_box()

        if median_size is not None:
            seed_size = math.sqrt(seed_image.shape[0] * seed_image.shape[1])
            scale = min(6.0, max(0.1, random.gauss(median_size / seed_size, 0.5)))
            seed_image = cv2.resize(seed_image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # Paint used regions
        mask = np.ones((rows, cols), dtype=np.uint8) * 255
        for (x1, y1, x2, y2) in bboxes:
            mask[int(y1) : int(y2), int(x1) : int(x2)] = 0

        mask = cv2.distanceTransform(mask, cv2.DIST_L2, 3, dstType=cv2.CV_32F)

        max_size = max(seed_image.shape[0], seed_image.shape[1])
        half_width = seed_image.shape[1] // 2
        half_height = seed_image.shape[0] // 2

        mask[: half_height + 1, :] = 0
        mask[:, : half_width + 1] = 0
        mask[mask.shape[0] - half_height - 1 :, :] = 0
        mask[:, mask.shape[1] - half_width - 1 :] = 0

        local_max = mask > max_size
        if not local_max.any():
            return {}

        nz_rows, nz_cols = np.nonzero(local_max)

        index = random.choice(np.arange(len(nz_rows)))

        x1 = nz_cols[index] - seed_image.shape[1] // 2
        y1 = nz_rows[index] - seed_image.shape[0] // 2
        x2 = x1 + seed_image.shape[1]
        y2 = y1 + seed_image.shape[0]

        return {
            "seed_image": seed_image,
            "seed_bbox": (x1, y1, x2, y2),
            "seed_p": ((x1 + x2) // 2, (y1 + y2) // 2),
            "seed_label": seed_label,
            "use_seamless_clone": self.seamless_clone_p > random.random(),
        }

    def apply(self, img, seed_image=None, seed_p=None, use_seamless_clone=False, seed_bbox=None, **params):
        if seed_image is not None:
            if use_seamless_clone:
                mask = np.ones(seed_image.shape[:2], dtype=np.uint8) * 255
                mask[0, :] = 0
                mask[:, 0] = 0
                mask[mask.shape[0] - 1, :] = 0
                mask[:, mask.shape[1] - 1] = 0

                return cv2.seamlessClone(
                    src=np.ascontiguousarray(seed_image),
                    dst=np.ascontiguousarray(img),
                    mask=mask,
                    p=seed_p,
                    flags=cv2.NORMAL_CLONE,
                )
            else:
                x1, y1, x2, y2 = seed_bbox
                img_hard = img.copy()
                img_hard[y1:y2, x1:x2] = seed_image
                return img_hard
        return img

    def apply_to_bboxes(self, bboxes, seed_bbox=None, seed_label=None, **params):
        if seed_bbox is not None:
            t = A.convert_bbox_to_albumentations(seed_bbox, "pascal_voc", params["rows"], params["cols"])
            bboxes = bboxes + [(*t, seed_label)]
        return bboxes

    def _select_box(self) -> Tuple[np.ndarray, int]:
        n = len(self.images)
        (index,) = random.choices(np.arange(n), self.sample_weights, k=1)

        image = self.read_image(self.images[index])
        x1, y1, x2, y2 = self.bboxes[index]

        roi = slice(y1, y2), slice(x1, x2)
        seed_image = image[roi]
        seed_label = self.labels[index]

        # Augment
        seed_image = self.transform(image=seed_image)["image"]

        return seed_image, seed_label

    def get_transform_init_args_names(self):
        return ()
