import random

import albumentations as A
import cv2
import numpy as np
import torch

__all__ = ["BoxesDropout", "CoarseDropoutWithBboxes"]


class BoxesDropout(A.DualTransform):
    """
    Remove objects and fill image & mask corresponding to removed bboxes.
    """

    def __init__(
        self,
        max_objects=5,
        max_fraction=0.2,
        image_fill_value=0,
        mask_fill_value=0,
        always_apply=False,
        drop_overlapping_boxes=True,
        overlap_iou=0.35,
        p=0.5,
    ):
        """
        Args:
            max_objects: Maximum number of labels that can be zeroed out. Can be tuple, in this case it's [min, max]
            image_fill_value: Fill value to use when filling image.
                Can be 'inpaint' to apply inpaining (works only  for 3-chahnel images)
            mask_fill_value: Fill value to use when filling mask.

        Targets:
            image, mask

        Image types:
            uint8, float32
        """
        super(BoxesDropout, self).__init__(always_apply, p)
        self.max_objects = max_objects
        self.max_fraction = max_fraction
        self.image_fill_value = image_fill_value
        self.mask_fill_value = mask_fill_value
        self.drop_overlapping_boxes = drop_overlapping_boxes
        self.overlap_iou = overlap_iou

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
        }

    def get_params_dependent_on_targets(self, params):
        from torchvision.ops import box_iou

        image = params["image"]
        rows, cols = image.shape[:2]
        bboxes = A.denormalize_bboxes(params["bboxes"], rows, cols)

        num_bboxes = len(bboxes)
        max_num_objects_to_drop = min(self.max_objects, int(self.max_fraction * num_bboxes))

        if max_num_objects_to_drop == 0:
            dropout_mask = None
            objects_to_drop = []
        else:
            indexes = np.arange(num_bboxes)

            objects_to_drop = random.randint(1, max_num_objects_to_drop)
            objects_to_drop = set(random.sample(indexes.tolist(), objects_to_drop))

            if self.drop_overlapping_boxes:
                bboxes = np.asarray([box[:4] for box in bboxes]).reshape((-1, 4))
                iou: np.ndarray = box_iou(torch.from_numpy(bboxes), torch.from_numpy(bboxes)).numpy()
                np.fill_diagonal(iou, 0)

                additional_objects_to_drop = []

                for obj in objects_to_drop:
                    overlapping = np.flatnonzero(iou[obj] > self.overlap_iou)
                    additional_objects_to_drop.extend(overlapping)

                objects_to_drop.union(set(additional_objects_to_drop))

                if len(objects_to_drop) > max_num_objects_to_drop:
                    # If the total number of objects to drop exceeds the allowed number,
                    # we revert skip dropout
                    objects_to_drop = []

            dropout_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.bool)
            for object_index, object_bbox in enumerate(bboxes):
                x_min, y_min, x_max, y_max = object_bbox[:4]
                if object_index in objects_to_drop:
                    dropout_mask[int(y_min) : int(y_max), int(x_min) : int(x_max)] = True

        return {"dropout_mask": dropout_mask, "objects_to_drop": objects_to_drop}

    def apply_to_bboxes(self, bboxes, **params):
        objects_to_drop = params["objects_to_drop"]
        return [box for i, box in enumerate(bboxes) if i not in objects_to_drop]

    def apply(self, img, dropout_mask=None, **params):
        if dropout_mask is None:
            return img

        if self.image_fill_value == "inpaint":
            dropout_mask = dropout_mask.astype(np.uint8)
            _, _, w, h = cv2.boundingRect(dropout_mask)
            radius = min(3, max(w, h) // 2)
            img = cv2.inpaint(img, dropout_mask, radius, cv2.INPAINT_NS)
        else:
            img = img.copy()
            img[dropout_mask] = self.image_fill_value

        return img

    def apply_to_mask(self, img, dropout_mask=None, **params):
        if dropout_mask is None:
            return img

        img = img.copy()
        img[dropout_mask] = self.mask_fill_value
        return img

    def get_transform_init_args_names(self):
        return ("max_objects", "image_fill_value", "mask_fill_value")


class CoarseDropoutWithBboxes(A.CoarseDropout):
    def __init__(
        self,
        uncertain_visibility: float = 0.75,
        uncertain_label: int = None,
        min_visibility=0.5,
        max_holes=8,
        max_height=8,
        max_width=8,
        min_holes=None,
        min_height=None,
        min_width=None,
        fill_value=0,
        mask_fill_value=None,
        always_apply=False,
        p=0.5,
    ):
        if uncertain_visibility and uncertain_label is None:
            raise ValueError()

        super().__init__(
            max_holes=max_holes,
            max_height=max_height,
            max_width=max_width,
            min_holes=min_holes,
            min_height=min_height,
            min_width=min_width,
            fill_value=fill_value,
            mask_fill_value=mask_fill_value,
            always_apply=always_apply,
            p=p,
        )
        self.uncertain_label = uncertain_label
        self.uncertain_visibility = uncertain_visibility
        self.min_visibility = min_visibility

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        bboxes = params["bboxes"]

        rows, cols = img.shape[:2]

        holes = []

        for _n in range(random.randint(self.min_holes, self.max_holes)):
            hole_height = random.randint(self.min_height, self.max_height)
            hole_width = random.randint(self.min_width, self.max_width)

            y1 = random.randint(0, rows - hole_height)
            x1 = random.randint(0, cols - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            holes.append((x1, y1, x2, y2))

        holes_mask = np.zeros((rows, cols), dtype=np.bool)
        for x1, y1, x2, y2 in holes:
            holes_mask[y1:y2, x1:x2] = True

        keep_bboxes = []
        new_labels = []

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2, label = A.denormalize_bbox(bbox, rows, cols)
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            bbox_mask = np.zeros((rows, cols), dtype=np.bool)
            bbox_mask[y1:y2, x1:x2] = True
            area_before = bbox_mask.sum()
            bbox_mask[holes_mask] = False
            area_after = bbox_mask.sum()
            visibility = area_after / area_before

            if visibility > self.min_visibility:
                if self.uncertain_visibility is not None and visibility < self.uncertain_visibility:
                    label = self.uncertain_label

            keep_bboxes.append(visibility > self.min_visibility)
            new_labels.append(label)

        return {"holes": holes, "keep_bboxes": keep_bboxes, "new_labels": new_labels}

    def apply_to_bboxes(self, bboxes, keep_bboxes=None, new_labels=None, **params):
        bboxes = [tuple(box[:4]) + tuple([label]) for box, label, keep in zip(bboxes, new_labels, keep_bboxes) if keep]
        return bboxes
