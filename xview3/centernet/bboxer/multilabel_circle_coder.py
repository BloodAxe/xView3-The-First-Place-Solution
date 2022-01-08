import math
from collections import namedtuple
from typing import Tuple, Optional, Union
from torch.nn import functional as F
import numpy as np
import torch
from torch import Tensor

from .functional import *

__all__ = [
    "MultilabelCircleNetCoder",
    "MultilabelCircleNetEncodeResult",
    "MultilabelCircleNetDecodeResult",
]

from .functional import draw_circle

MultilabelCircleNetEncodeResult = namedtuple(
    "MultilabelCircleNetEncodeResult",
    ["heatmap", "is_fishing", "is_vessel", "lengths", "offset"],
)
MultilabelCircleNetDecodeResult = namedtuple(
    "MultilabelCircleNetDecodeResult",
    ["centers", "lengths", "is_fishing", "is_vessel", "scores"],
)


def cast_to_float_if_on_cpu(x: Tensor) -> Tensor:
    if x.device == torch.device("cpu") and x.dtype == torch.float16:
        x = x.float()
    return x


class MultilabelCircleNetCoder:
    output_stride: int
    max_objects: int
    heatmap_encoding: str
    labels_encoding: str
    ignore_value: Optional[int]
    image_height: int
    image_width: int
    image_size: Tuple[int, int]
    fixed_radius: int
    labels_radius: int

    def __init__(
        self,
        image_size: Tuple[int, int],
        output_stride: int,
        max_objects: int,
        heatmap_encoding: str,
        labels_encoding: str,
        ignore_value: Optional[int],
        fixed_radius: Optional[int] = None,
        labels_radius: Optional[int] = None,
        nms_kernel_size: int = 3,
        nms_method=centernet_heatmap_nms,
        ignore_low_confidence_detections: bool = False,
    ):
        image_size = image_size[:2]

        if heatmap_encoding not in {"umich", "msra"}:
            raise KeyError(heatmap_encoding)

        if labels_encoding not in {"point", "heatmap", "circle", "circle_ignore"}:
            raise KeyError(labels_encoding)

        self.heatmap_encoding = heatmap_encoding
        self.labels_encoding = labels_encoding
        self.output_stride = output_stride
        self.image_size = image_size
        self.image_height, self.image_width = image_size
        self.max_objects = max_objects
        self.ignore_value = ignore_value
        self.fixed_radius = fixed_radius
        self.labels_radius = labels_radius
        self.nms_kernel_size = nms_kernel_size
        self.ignore_low_confidence_detections = ignore_low_confidence_detections
        self.nms_method = nms_method

    def __repr__(self):
        return (
            "MultilabelCircleNetCoder("
            + f"image_size={self.image_size}, "
            + f"stride={self.output_stride}, "
            + f"max_objects={self.max_objects}, "
            + f"labels_encoding={self.labels_encoding}, "
            + f"heatmap_encoding={self.heatmap_encoding}, "
            + f"fixed_radius={self.fixed_radius}, "
            + f"ignore_index={self.ignore_value}, "
            + f"ignore_low_confidence_detections={self.ignore_low_confidence_detections}"
            + ")"
        )

    def box_coder_for_image_size(self, image_size):
        image_size = image_size[:2]
        if self.image_height == image_size[0] and self.image_width == image_size[1]:
            return self
        return MultilabelCircleNetCoder(
            image_size=image_size,
            output_stride=self.output_stride,
            max_objects=self.max_objects,
            heatmap_encoding=self.heatmap_encoding,
            labels_encoding=self.labels_encoding,
            ignore_value=self.ignore_value,
            fixed_radius=self.fixed_radius,
            labels_radius=self.labels_radius,
            nms_kernel_size=self.nms_kernel_size,
            nms_method=self.nms_method,
            ignore_low_confidence_detections=self.ignore_low_confidence_detections,
        )

    def encode(
        self,
        centers: np.ndarray,
        confidences: np.ndarray,
        is_vessel: np.ndarray,
        is_fishing: np.ndarray,
        lengths: np.ndarray,
    ) -> MultilabelCircleNetEncodeResult:
        """
        :param centers [N,2] (x1,y1)
        :param lengths [N] (Ship length in pixels, See XView3DataModule.encode_lengths)
        """

        if (
            len(centers) != len(confidences)
            or len(centers) != len(is_vessel)
            or len(centers) != len(is_fishing)
            or len(centers) != len(lengths)
        ):
            raise RuntimeError(
                f"Got inconsistent length of centers ({centers.shape}), lengths ({lengths.shape}), is_vessel ({is_vessel.shape}) and is_fishing ({is_fishing.shape})"
            )

        # Sort objects by confidence
        low_conf_mask = confidences == "LOW"
        med_conf_mask = confidences == "MEDIUM"
        hgh_conf_mask = confidences == "HIGH"
        order = [low_conf_mask, med_conf_mask, hgh_conf_mask]

        def rearrange(x, masks, lengths):
            sorted_sub_arrays = []
            for m in masks:
                subsampled_lengths = lengths[m]
                subsampled_elements = x[m]
                order = np.argsort(-subsampled_lengths)
                sorted_sub_arrays.append(subsampled_elements[order])
            return np.concatenate(sorted_sub_arrays, axis=0)

        centers = rearrange(centers, order, lengths)
        confidences = rearrange(confidences, order, lengths)
        is_vessel = rearrange(is_vessel, order, lengths)
        is_fishing = rearrange(is_fishing, order, lengths)
        lengths = rearrange(lengths, order, lengths)

        # Reconstruct LOW mask after reordering
        low_conf_mask = confidences == "LOW"

        output_height = self.image_height // self.output_stride
        output_width = self.image_width // self.output_stride

        heatmap = np.zeros((1, output_height, output_width), dtype=np.float32)

        is_vessel_map = np.zeros((1, output_height, output_width), dtype=np.float32)
        is_fishing_map = np.zeros((1, output_height, output_width), dtype=np.float32)

        length_map = np.zeros((1, output_height, output_width), dtype=np.float32)
        offset_map = np.zeros((2, output_height, output_width), dtype=np.float32)

        num_objs = len(centers)

        centers_scaled = centers.astype(np.float32) / float(self.output_stride)
        lengths_scaled = lengths.astype(np.float32) / float(self.output_stride)

        log_lengths = self.length2log(lengths)  # Transform targets to log-space

        for i in range(num_objs):
            center = centers_scaled[i]
            center_int = center.astype(int)

            if self.fixed_radius:
                heatmap_radius = self.fixed_radius
            else:
                heatmap_radius = max(2, int(math.ceil(lengths_scaled[i] * 0.5)))

            if self.heatmap_encoding == "umich":
                if low_conf_mask[i] and self.ignore_low_confidence_detections:
                    draw_circle(heatmap[0], center_int, heatmap_radius, self.ignore_value)
                else:
                    draw_umich_gaussian(heatmap[0], center_int, heatmap_radius)
            elif self.heatmap_encoding == "point":
                if low_conf_mask[i] and self.ignore_low_confidence_detections:
                    heatmap[0, center_int[1], center_int[0]] = self.ignore_value
                else:
                    heatmap[0, center_int[1], center_int[0]] = 1
            else:
                raise KeyError(self.heatmap_encoding)

            if self.labels_encoding == "point":
                is_vessel_map[0, center_int[1], center_int[0]] = is_vessel[i]
                is_fishing_map[0, center_int[1], center_int[0]] = is_fishing[i]
                length_map[0, center_int[1], center_int[0]] = log_lengths[i]
            elif self.labels_encoding == "circle":
                if self.labels_radius:
                    labels_radius = self.labels_radius
                else:
                    labels_radius = heatmap_radius

                draw_circle(is_vessel_map[0], center_int, labels_radius, is_vessel[i])
                draw_circle(is_fishing_map[0], center_int, labels_radius, is_fishing[i])
                draw_circle(length_map[0], center_int, labels_radius, log_lengths[i])

            else:
                raise KeyError(self.labels_encoding)

            offset_map[0, center_int[1], center_int[0]] = center[0] - center_int[0]
            offset_map[1, center_int[1], center_int[0]] = center[1] - center_int[1]

        return MultilabelCircleNetEncodeResult(
            heatmap=heatmap,
            is_vessel=is_vessel_map,
            is_fishing=is_fishing_map,
            lengths=length_map,
            offset=offset_map,
        )

    @torch.no_grad()
    def decode(
        self,
        objectness_map: Tensor,
        is_vessel_map: Tensor,
        is_fishing_map: Tensor,
        length_map: Tensor,
        offset_map: Tensor,
        max_objects: Optional[int] = None,
        apply_activation: bool = False,
    ) -> MultilabelCircleNetDecodeResult:
        """
        Decode CenterNet predictions
        :param objectness_map: [B, 1, H, W]
        :param is_vessel_map: [B, 1, H, W]
        :param is_fishing_map: [B, 1, H, W]
        :param length_map: [B, 1, H, W]
        :param offset_map: [B, 2, H, W]
        :param max_objects: Maximum number of objects (K)
        :param apply_activation:
        :return: Tuple of 4 elements (centers, lengths, labels, scores)
            - [B, K, 2]
            - [B, K]
            - [B, K]
            - [B, K]
        """
        batch, _, height, width = objectness_map.size()

        objectness_map = cast_to_float_if_on_cpu(objectness_map)
        is_fishing_map = cast_to_float_if_on_cpu(is_fishing_map)
        is_vessel_map = cast_to_float_if_on_cpu(is_vessel_map)

        length_map = cast_to_float_if_on_cpu(length_map)

        if apply_activation:
            objectness_map = objectness_map.sigmoid()
            is_fishing_map = is_fishing_map.sigmoid()
            is_vessel_map = is_vessel_map.sigmoid()

        if max_objects is None:
            max_objects = self.max_objects

        # This is necessary if we use non-masked image (with NaNs)
        objectness_map = torch.masked_fill(objectness_map, ~torch.isfinite(objectness_map), 0)
        is_fishing_map = torch.masked_fill(is_fishing_map, ~torch.isfinite(is_fishing_map), 0)
        is_vessel_map = torch.masked_fill(is_vessel_map, ~torch.isfinite(is_vessel_map), 0)

        obj_map = self.nms_method(objectness_map, kernel=self.nms_kernel_size)

        # Limit K to prevent having K more W * H
        max_objects = min(max_objects, height * width)

        # Decode centers
        obj_scores, inds, _, ys, xs = centernet_topk(obj_map, top_k=max_objects)
        if offset_map is not None:
            offset_map = cast_to_float_if_on_cpu(offset_map)
            offset_map = centernet_tranpose_and_gather_feat(offset_map, inds)
            xs = xs.view(batch, max_objects, 1) + offset_map[:, :, 0:1]
            ys = ys.view(batch, max_objects, 1) + offset_map[:, :, 1:2]
        elif self.output_stride == 1:
            xs = xs.view(batch, max_objects, 1)
            ys = ys.view(batch, max_objects, 1)
        else:
            xs = xs.view(batch, max_objects, 1) + 0.5
            ys = ys.view(batch, max_objects, 1) + 0.5
        pred_centers = torch.cat([xs, ys], dim=2) * self.output_stride

        # Decode lengths
        pred_lengths = centernet_tranpose_and_gather_feat(length_map, inds)
        pred_lengths = pred_lengths[:, :, 0]  # Squeeze length

        # Decode is_vessel
        is_vessel, _ = centernet_tranpose_and_gather_feat(is_vessel_map, inds).max(dim=2)

        # Decode is_fishing
        is_fishing, _ = centernet_tranpose_and_gather_feat(is_fishing_map, inds).max(dim=2)

        return MultilabelCircleNetDecodeResult(
            centers=pred_centers,
            lengths=self.log2length(pred_lengths),  # Since length are in log-space, convert back to linear units
            is_fishing=is_fishing,
            is_vessel=is_vessel,
            scores=obj_scores,
        )

    @classmethod
    def log2length(cls, x: Union[Tensor]) -> Union[Tensor]:
        if torch.is_tensor(x):
            return torch.exp(F.relu(x)) - 1
        else:
            return np.exp(np.clip(x, a_min=0, a_max=None)) - 1

    @classmethod
    def length2log(cls, x: np.ndarray) -> np.ndarray:
        return np.log(x + 1)
