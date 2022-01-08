from functools import partial
from typing import Optional, List, Callable, Union, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_toolbelt.inference import tta, ImageSlicer, TileMerger
from pytorch_toolbelt.inference.ensembling import ApplySigmoidTo, Ensembler, ApplySoftmaxTo
from pytorch_toolbelt.utils import image_to_tensor
from pytorch_toolbelt.utils.distributed import get_rank
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from ..bboxer import MultilabelCircleNetCoder
from ..constants import (
    CENTERNET_OUTPUT_CLASS_MAP,
    CENTERNET_OUTPUT_OBJECTNESS_MAP,
    CENTERNET_OUTPUT_OFFSET,
    CENTERNET_OUTPUT_SIZE,
    CENTERNET_OUTPUT_VESSEL_MAP,
    CENTERNET_OUTPUT_FISHING_MAP,
)

__all__ = [
    "centernet_d4_tta",
    "centernet_d2_tta",
]


def move_to_device_non_blocking(x: Tensor, device: torch.device):
    if x.device != device:
        x = x.to(device=device, non_blocking=True)
    return x


def centernet_flips_tta(model: nn.Module, average_heatmap=True):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return tta.GeneralizedTTA(
        model,
        augment_fn=tta.flips_image_augment,
        deaugment_fn={
            CENTERNET_OUTPUT_OBJECTNESS_MAP: partial(tta.flips_image_deaugment, reduction="mean" if average_heatmap else "sum"),
            CENTERNET_OUTPUT_CLASS_MAP: partial(tta.flips_image_deaugment, reduction="mean" if average_heatmap else "sum"),
            CENTERNET_OUTPUT_SIZE: tta.flips_image_deaugment,
            CENTERNET_OUTPUT_OFFSET: tta.flips_image_deaugment,
        },
    )


def centernet_d2_tta(model: nn.Module, average_heatmap=True):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return tta.GeneralizedTTA(
        model,
        augment_fn=tta.d2_image_augment,
        deaugment_fn={
            CENTERNET_OUTPUT_OBJECTNESS_MAP: partial(tta.d2_image_deaugment, reduction="mean" if average_heatmap else "sum"),
            CENTERNET_OUTPUT_CLASS_MAP: partial(tta.d2_image_deaugment, reduction="mean" if average_heatmap else "sum"),
            CENTERNET_OUTPUT_SIZE: tta.d2_image_deaugment,
            CENTERNET_OUTPUT_OFFSET: tta.d2_image_deaugment,
        },
    )


def centernet_d4_tta(model: nn.Module, average_heatmap=True):
    return tta.GeneralizedTTA(
        model,
        augment_fn=tta.d4_image_augment,
        deaugment_fn={
            CENTERNET_OUTPUT_OBJECTNESS_MAP: partial(tta.d4_image_deaugment, reduction="mean" if average_heatmap else "sum"),
            CENTERNET_OUTPUT_CLASS_MAP: partial(tta.d4_image_deaugment, reduction="mean" if average_heatmap else "sum"),
            CENTERNET_OUTPUT_SIZE: tta.d4_image_deaugment,
            CENTERNET_OUTPUT_OFFSET: tta.d4_image_deaugment,
        },
    )


def centernet_ms_size_deaugment(
    images: List[Tensor],
    size_offsets: List[Union[int, Tuple[int, int]]],
    reduction: Optional[Union[str, Callable]] = "mean",
    mode: str = "bilinear",
    align_corners: bool = True,
    stride: int = 1,
) -> Tensor:
    if len(images) != len(size_offsets):
        raise ValueError("Number of images must be equal to number of size offsets")

    deaugmented_outputs = []
    for image, offset in zip(images, size_offsets):
        batch_size, channels, rows, cols = image.size()
        original_size = rows - offset // stride, cols - offset // stride
        scaled_image = torch.nn.functional.interpolate(image, size=original_size, mode=mode, align_corners=align_corners)
        size_scale = torch.tensor(
            [original_size[0] / rows, original_size[1] / cols],
            dtype=scaled_image.dtype,
            device=scaled_image.device,
        ).view((1, 2, 1, 1))

        deaugmented_outputs.append(scaled_image * size_scale)

    deaugmented_outputs = torch.stack(deaugmented_outputs)
    if reduction == "mean":
        deaugmented_outputs = deaugmented_outputs.mean(dim=0)
    if reduction == "sum":
        deaugmented_outputs = deaugmented_outputs.sum(dim=0)
    if callable(reduction):
        deaugmented_outputs = reduction(deaugmented_outputs, dim=0)

    return deaugmented_outputs


def real_model(model: nn.Module):
    if isinstance(model, (ApplySigmoidTo, ApplySoftmaxTo)):
        return real_model(model.model)
    if isinstance(model, (tta.GeneralizedTTA, tta.MultiscaleTTA)):
        return real_model(model.model)
    if isinstance(model, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return real_model(model.module)

    return model


def centernet_ms_tta(model: nn.Module, size_offsets: List[int], average_heatmap=True):
    output_stride = real_model(model).output_stride

    return tta.MultiscaleTTA(
        model,
        size_offsets,
        deaugment_fn={
            CENTERNET_OUTPUT_OBJECTNESS_MAP: partial(
                tta.ms_image_deaugment,
                reduction="mean" if average_heatmap else "sum",
                stride=output_stride,
            ),
            CENTERNET_OUTPUT_CLASS_MAP: partial(
                tta.ms_image_deaugment,
                reduction="mean" if average_heatmap else "sum",
                stride=output_stride,
            ),
            CENTERNET_OUTPUT_SIZE: partial(centernet_ms_size_deaugment, stride=output_stride),
            CENTERNET_OUTPUT_OFFSET: partial(tta.ms_image_deaugment, stride=output_stride),
        },
    )


def get_box_coder_from_model(model):
    if isinstance(model, nn.DataParallel):
        return get_box_coder_from_model(model.module)

    if isinstance(model, (tta.GeneralizedTTA, tta.MultiscaleTTA)):
        return get_box_coder_from_model(model.model)

    if isinstance(model, ApplySigmoidTo):
        return get_box_coder_from_model(model.model)

    if isinstance(model, Ensembler):
        return get_box_coder_from_model(model.models[0])

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return get_box_coder_from_model(model.module)
    return model.box_coder


class SlicedIterableDataset(IterableDataset):
    def __init__(self, image, image_slices, predictions_slicer):
        self.dataset = SlicedDataset(image, image_slices, predictions_slicer)

    def __iter__(self):
        for index in range(len(self.dataset)):
            sample = self.dataset[index]
            if not sample[0].eq(0).all():
                yield sample


class SlicedDataset(Dataset):
    def __init__(self, image, image_slices, predictions_slicer):
        self.image = image
        self.image_slices = image_slices
        self.predictions_slicer = predictions_slicer

    def __len__(self):
        return len(self.image_slices.crops)

    def __getitem__(self, index):
        image_coords, pred_coords = (
            self.image_slices.crops[index],
            self.predictions_slicer.crops[index],
        )

        x, y, tile_width, tile_height = image_coords

        x1 = x
        y1 = y
        x2 = x + tile_width
        y2 = y + tile_height

        tile = self.image[y1:y2, x1:x2]
        assert tile.shape[0] == tile_height
        assert tile.shape[1] == tile_width

        # tile_normalized = self.normalization(image=tile)["image"]
        tile_normalized = tile.copy()
        tile_normalized[~np.isfinite(tile_normalized)] = 0
        image = image_to_tensor(tile_normalized)
        return image, image_coords, pred_coords


@torch.no_grad()
def multilabel_centernet_tiled_inference(
    model: nn.Module,
    image: np.ndarray,
    box_coder: MultilabelCircleNetCoder,
    tile_size=1024,
    tile_step=768,
    fp16=True,
    tile_weight="pyramid",
    accumulate_on_gpu=False,
    batch_size=1,
    channels_last=False,
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    original_shape = image.shape[:2]

    pad_op = A.PadIfNeeded(
        min_width=None,
        min_height=None,
        pad_width_divisor=tile_size,
        pad_height_divisor=tile_size,
        position=A.PadIfNeeded.PositionType.TOP_LEFT,
        value=0,
        border_mode=cv2.BORDER_CONSTANT,
    )
    image = pad_op(image=image)["image"]

    padded_shape = image.shape[:2]
    downsample = box_coder.output_stride

    assert tile_size % downsample == 0
    assert tile_step % downsample == 0
    assert padded_shape[0] % downsample == 0
    assert padded_shape[1] % downsample == 0

    predictions_shape = padded_shape[0] // downsample, padded_shape[1] // downsample

    image_slices = ImageSlicer(
        padded_shape,
        tile_size=tile_size,
        tile_step=tile_step,
        weight=tile_weight,
        image_margin=(0, 0, 0, 0),
    )
    predictions_slicer = ImageSlicer(
        predictions_shape,
        tile_size=tile_size // downsample,
        tile_step=tile_step // downsample,
        weight=tile_weight,
        image_margin=(0, 0, 0, 0),
    )

    tile_device = "cuda" if accumulate_on_gpu else "cpu"
    tile_dtype = (torch.float16 if fp16 else torch.float32) if accumulate_on_gpu else torch.float32

    heatmap_merger = TileMerger(
        predictions_slicer.target_shape,
        channels=1,
        weight=predictions_slicer.weight,
        dtype=tile_dtype,
        device=tile_device,
    )
    is_vessel_merger = TileMerger(
        predictions_slicer.target_shape,
        channels=1,
        weight=predictions_slicer.weight,
        dtype=tile_dtype,
        device=tile_device,
    )
    is_fishing_merger = TileMerger(
        predictions_slicer.target_shape,
        channels=1,
        weight=predictions_slicer.weight,
        dtype=tile_dtype,
        device=tile_device,
    )
    offset_merger = TileMerger(
        predictions_slicer.target_shape,
        channels=2,
        weight=predictions_slicer.weight,
        dtype=tile_dtype,
        device=tile_device,
    )
    size_merger = TileMerger(
        predictions_slicer.target_shape,
        channels=1,
        weight=predictions_slicer.weight,
        dtype=tile_dtype,
        device=tile_device,
    )

    if len(image_slices.crops) != len(predictions_slicer.crops):
        raise RuntimeError("Number of slices in images does not equal to number of predictions slicer")

    has_offset_predictions = False
    dataset = SlicedIterableDataset(image, image_slices, predictions_slicer)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    for image_tensor, image_coords, pred_coords in tqdm(loader, position=get_rank()):
        image_tensor = image_tensor.to(
            device, non_blocking=True, memory_format=torch.channels_last if channels_last else torch.contiguous_format
        )
        with torch.cuda.amp.autocast(fp16):
            output = model(image_tensor)

        objectness = move_to_device_non_blocking(output[CENTERNET_OUTPUT_OBJECTNESS_MAP], heatmap_merger.device)
        vessel = move_to_device_non_blocking(output[CENTERNET_OUTPUT_VESSEL_MAP], is_vessel_merger.device)
        fishing = move_to_device_non_blocking(output[CENTERNET_OUTPUT_FISHING_MAP], is_fishing_merger.device)
        size = move_to_device_non_blocking(output[CENTERNET_OUTPUT_SIZE], size_merger.device)
        if CENTERNET_OUTPUT_OFFSET in output:
            offset = move_to_device_non_blocking(output[CENTERNET_OUTPUT_OFFSET], offset_merger.device)
            has_offset_predictions = True

        if not accumulate_on_gpu:
            torch.cuda.synchronize()

        heatmap_merger.integrate_batch(objectness, pred_coords)
        is_vessel_merger.integrate_batch(vessel, pred_coords)
        is_fishing_merger.integrate_batch(fishing, pred_coords)
        size_merger.integrate_batch(size, pred_coords)
        if CENTERNET_OUTPUT_OFFSET in output:
            offset_merger.integrate_batch(offset, pred_coords)

    outputs = {
        CENTERNET_OUTPUT_OBJECTNESS_MAP: heatmap_merger.merge().unsqueeze(0),
        CENTERNET_OUTPUT_VESSEL_MAP: is_vessel_merger.merge().unsqueeze(0),
        CENTERNET_OUTPUT_FISHING_MAP: is_fishing_merger.merge().unsqueeze(0),
        CENTERNET_OUTPUT_SIZE: size_merger.merge().unsqueeze(0),
    }
    if has_offset_predictions:
        outputs[CENTERNET_OUTPUT_OFFSET] = offset_merger.merge().unsqueeze(0)
    return outputs
