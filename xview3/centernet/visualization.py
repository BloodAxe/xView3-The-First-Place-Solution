import albumentations as A
import cv2
import numpy as np
import torch
from pytorch_toolbelt.datasets import INPUT_IMAGE_KEY, INPUT_INDEX_KEY
from pytorch_toolbelt.utils import (
    to_numpy,
    vstack_header,
    hstack_autopad,
)

from .bboxer import *
from .constants import *
from ..dataset import XView3DataModule
from ..constants import (
    TARGET_CENTERS_KEY,
    TARGET_LENGTHS_KEY,
    INPUT_SCENE_ID_KEY,
    INPUT_SCENE_CROP_KEY,
    PIX_TO_M,
    INPUT_FOLDER_KEY,
    IGNORE_LABEL,
    TARGET_FISHING_KEY,
    TARGET_VESSEL_KEY,
)

# USE AS FOLOWS: colors[is_vessel][is_fishing]


OBJECTS_COLOR = {
    # Is Vessel - False
    False: {
        # Is Vessel, Non-Fishing (Default) - pink color
        False: (203, 192, 255),
        # Is Vessel, Is-Fishing, May happen only during prediction
        True: (203, 192, 255),
        # Non-vessel, fishing status unknown
        IGNORE_LABEL: (203, 192, 255),
    },
    # Is Vessel - True
    True: {
        # Non-fishing vessels are mint-green
        False: (52, 255, 52),
        # Fishing vessels are orange
        True: (0, 127, 255),
        # Vessel with unknown fishing status - Purple
        IGNORE_LABEL: (255, 0, 255),
    },
    # Unknown platforms are gray
    IGNORE_LABEL: {True: (127, 127, 127), False: (127, 127, 127), IGNORE_LABEL: (127, 127, 127)},
}


def get_flatten_object_colors():
    is_vessel, is_fishing = XView3DataModule.decode_labels(np.arange(3))
    colors = []
    for class_idx in range(3):
        color = OBJECTS_COLOR[is_vessel[class_idx]][is_fishing[class_idx]]
        color = np.array(color, dtype=float).reshape((1, 1, 3))
        colors.append(color)

    return colors


FLATTEN_OBJECT_COLORS = get_flatten_object_colors()


def visualize_titles(img, center, radius, title, text_color=(255, 255, 255), font_thickness=1, font_scale=0.5):
    cx, cy = center
    ((text_width, text_height), _) = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    cv2.putText(
        img,
        title,
        (cx - text_width // 2, cy - radius - int(0.3 * text_height)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        font_thickness,
        lineType=cv2.LINE_AA,
    )
    return img


def vis_detections_opencv(
    img,
    centers,
    lengths,
    is_vessel_vec,
    is_fishing_vec,
    scores,
    is_vessel_probs=None,
    is_fishing_probs=None,
    show_title=True,
    font_scale=0.5,
    thickness=2,
    alpha=0.25,
):
    overlay = img.copy()
    num_detections = len(centers)

    for i in range(num_detections):
        cx, cy = centers[i]
        cx = int(cx)
        cy = int(cy)

        length = lengths[i]
        is_vessel = is_vessel_vec[i]
        is_fishing = is_fishing_vec[i]
        color = OBJECTS_COLOR[is_vessel][is_fishing]
        center = cx, cy
        radius = min(max(2, int(length * 0.5)), 100000)

        if length == 0:
            cv2.rectangle(
                overlay,
                pt1=(cx - 4, cy - 4),
                pt2=(cx + 4, cy + 4),
                color=color,
                thickness=thickness,
            )
        else:
            cv2.circle(overlay, center, radius, color=color, thickness=thickness, lineType=cv2.LINE_AA)

        if show_title:
            is_vessel_str = {0: "P", False: "P", 1: "V", True: "V", IGNORE_LABEL: "U"}[is_vessel]
            is_fishing_str = {0: "", False: "", 1: "/F", True: "/F", IGNORE_LABEL: "/U"}[is_fishing]

            if is_vessel_probs is not None:
                is_vessel_str = " {0}{1:.2f}".format(is_vessel_str, is_vessel_probs[i])

            if is_fishing_probs is not None:
                is_fishing_str = " {0}{1:.2f}".format(is_fishing_str, is_fishing_probs[i])

            caption = "{0}{1}".format(is_vessel_str, is_fishing_str)
            if scores is not None:
                score = " {0:.2f}".format(scores[i])
                caption += score

            visualize_titles(
                overlay,
                center=center,
                radius=radius,
                title=caption,
                font_scale=font_scale,
                text_color=color,
            )

    img = cv2.addWeighted(img, alpha, overlay, 1 - alpha, 0)
    return img


def create_false_color_composite(vv, vh, cmap=cv2.COLORMAP_OCEAN):
    """
    Returns a false color composite image of SAR bands for visualization.

    Returns:
        np.array: image (H, W, 3) ready for visualization
    """

    mean = (vv + vh) * 0.5

    mean8 = cv2.normalize(mean, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    mean8_rgb = cv2.cvtColor(mean8, cv2.COLOR_GRAY2RGB)
    mean8_cmap = cv2.applyColorMap(mean8, cmap)

    return ((mean8_cmap.astype(int) + mean8_rgb.astype(int)) // 2).astype(np.uint8)


def pseudocolor_float_image(x):
    mean8 = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return cv2.applyColorMap(mean8, cv2.COLORMAP_OCEAN)


def sar_to_rgb(sar_image):
    return create_false_color_composite(to_numpy(sar_image[0, ...]), to_numpy(sar_image[1, ...]))


@torch.no_grad()
def visualize_predictions_multilabel_centernet(
    batch,
    output,
    box_coder: MultilabelCircleNetCoder,
    alpha=0.35,
    max_image_size=None,
    min_confidence_score=0.5,
):
    images = []

    batch_size = batch[INPUT_IMAGE_KEY].size(0)
    image_size = batch[INPUT_IMAGE_KEY].size(2), batch[INPUT_IMAGE_KEY].size(3)
    half_size = batch[INPUT_IMAGE_KEY].size(2) // 2, batch[INPUT_IMAGE_KEY].size(3) // 2
    half_size_width_height = half_size[1], half_size[0]

    box_coder = box_coder.box_coder_for_image_size(image_size)

    gt_heatmap = to_numpy(batch[CENTERNET_TARGET_OBJECTNESS_MAP].detach().cpu().float())
    gt_vessel = to_numpy(batch[CENTERNET_TARGET_VESSEL_MAP].detach().cpu().float())
    gt_fishing = to_numpy(batch[CENTERNET_TARGET_FISHING_MAP].detach().cpu().float())

    pred_heatmap = output[CENTERNET_OUTPUT_OBJECTNESS_MAP].detach().float().sigmoid()
    pred_vessel = output[CENTERNET_OUTPUT_VESSEL_MAP].detach().float().sigmoid()
    pred_fishing = output[CENTERNET_OUTPUT_FISHING_MAP].detach().float().sigmoid()

    # Decode
    decoded = box_coder.decode(
        objectness_map=pred_heatmap,
        is_vessel_map=pred_vessel,
        is_fishing_map=pred_fishing,
        length_map=output[CENTERNET_OUTPUT_SIZE],
        offset_map=output.get(CENTERNET_OUTPUT_OFFSET, None),
        apply_activation=False,
    )

    pred_heatmap = to_numpy(pred_heatmap)
    pred_vessel = to_numpy(pred_vessel)
    pred_fishing = to_numpy(pred_fishing)

    for sample_idx in range(batch_size):
        image_orig = sar_to_rgb(batch[INPUT_IMAGE_KEY][sample_idx])

        scene_id = batch[INPUT_SCENE_ID_KEY][sample_idx]
        folder = batch[INPUT_FOLDER_KEY][sample_idx]
        sample_index_in_dataset = batch[INPUT_INDEX_KEY][sample_idx]
        crop_coords = batch[INPUT_SCENE_CROP_KEY][sample_idx]

        gt_heatmap_rgb = pseudocolor_heatmap(gt_heatmap[sample_idx, 0])
        gt_classmap_rgb = pseudocolor_vessel_fishing(gt_vessel[sample_idx, 0], gt_fishing[sample_idx, 0])

        pred_heatmap_rgb = pseudocolor_heatmap(pred_heatmap[sample_idx, 0])
        pred_classmap_rgb = pseudocolor_vessel_fishing(
            pred_vessel[sample_idx, 0] * pred_heatmap[sample_idx, 0],
            pred_fishing[sample_idx, 0] * pred_heatmap[sample_idx, 0],
        )

        true_vessel = np.array(batch[TARGET_VESSEL_KEY][sample_idx]).reshape((-1))
        true_fishing = np.array(batch[TARGET_FISHING_KEY][sample_idx]).reshape((-1))
        true_centers = np.array(batch[TARGET_CENTERS_KEY][sample_idx]).reshape((-1, 2))
        true_lengths = np.array(batch[TARGET_LENGTHS_KEY][sample_idx]).reshape((-1))

        if len(true_centers):
            gt_image = vis_detections_opencv(
                image_orig,
                centers=true_centers,
                lengths=XView3DataModule.decode_lengths(true_lengths) / PIX_TO_M,
                is_vessel_vec=true_vessel,
                is_fishing_vec=true_fishing,
                scores=None,
                show_title=True,
                alpha=alpha,
            )
        else:
            gt_image = image_orig

        gt_image = np.row_stack(
            [
                gt_image,
                np.column_stack(
                    [
                        cv2.resize(gt_heatmap_rgb, half_size_width_height, interpolation=cv2.INTER_CUBIC),
                        cv2.resize(gt_classmap_rgb, half_size_width_height, interpolation=cv2.INTER_CUBIC),
                    ]
                ),
            ]
        )

        pred_mask = decoded.scores[sample_idx] > min_confidence_score
        pred_centers = to_numpy(decoded.centers[sample_idx, pred_mask, :])
        pred_is_vessel_probs = to_numpy(decoded.is_vessel[sample_idx, pred_mask])
        pred_is_fishing_probs = to_numpy(decoded.is_fishing[sample_idx, pred_mask])

        pred_is_vessel = to_numpy(pred_is_vessel_probs > 0.5)
        pred_is_fishing = to_numpy(pred_is_fishing_probs > 0.5)

        pred_scores = to_numpy(decoded.scores[sample_idx, pred_mask])
        pred_lengths = to_numpy(decoded.lengths[sample_idx, pred_mask])

        if len(pred_centers):
            pred_image = vis_detections_opencv(
                image_orig,
                centers=pred_centers,
                lengths=XView3DataModule.decode_lengths(pred_lengths) / PIX_TO_M,
                is_vessel_vec=pred_is_vessel,
                is_fishing_vec=pred_is_fishing,
                is_vessel_probs=pred_is_vessel_probs,
                is_fishing_probs=pred_is_fishing_probs,
                scores=pred_scores,
                show_title=True,
                alpha=alpha,
            )
        else:
            pred_image = image_orig

        pred_image = np.row_stack(
            [
                pred_image,
                np.column_stack(
                    [
                        cv2.resize(pred_heatmap_rgb, half_size_width_height, interpolation=cv2.INTER_CUBIC),
                        cv2.resize(pred_classmap_rgb, half_size_width_height, interpolation=cv2.INTER_CUBIC),
                    ]
                ),
            ]
        )

        image_extra = []
        if batch[INPUT_IMAGE_KEY].size(1) > 2:
            for channel in range(2, int(batch[INPUT_IMAGE_KEY].size(1))):
                image_extra.append(pseudocolor_float_image(to_numpy(batch[INPUT_IMAGE_KEY][sample_idx, channel, ...])))

        image = vstack_header(
            hstack_autopad(image_extra + [gt_image, pred_image]),
            f"{sample_index_in_dataset} {folder}/{scene_id} {crop_coords} {image_size}",
        )

        # For tensorboard image must be converted to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if max_image_size is not None and max(*image.shape) > max_image_size:
            image = A.longest_max_size(image, max_image_size, cv2.INTER_CUBIC)

        images.append(image)

    return images


def pseudocolor_vessel_fishing(is_vessel, is_fishing):
    is_vessel = np.squeeze(to_numpy(is_vessel))
    is_fishing = np.squeeze(to_numpy(is_fishing))

    bgr = np.zeros((is_vessel.shape[0], is_vessel.shape[1], 3), dtype=np.uint8)
    bgr[...] = 40

    is_vessel_ignore = is_vessel == IGNORE_LABEL
    is_fishing_ignore = is_fishing == IGNORE_LABEL

    bgr[..., 0] = (~is_vessel_ignore * (1 - is_vessel) * 255).astype(np.uint8)
    bgr[..., 1] = (~is_vessel_ignore * is_vessel * 255).astype(np.uint8)
    bgr[..., 2] = (~is_fishing_ignore * is_fishing * 255).astype(np.uint8)
    bgr[is_vessel_ignore & is_fishing_ignore, :] = 0

    return bgr


def pseudocolor_heatmap(hm):
    hm = np.squeeze(to_numpy(hm))
    mask = np.isfinite(hm)
    ignored_mask = hm == IGNORE_LABEL

    hm = (hm * 255).astype(np.uint8)

    cmap = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    cmap[~mask] = 0
    cmap[ignored_mask] = (127, 127, 127)
    return cmap


__all__ = [
    "pseudocolor_heatmap",
    "visualize_predictions_multilabel_centernet",
]
