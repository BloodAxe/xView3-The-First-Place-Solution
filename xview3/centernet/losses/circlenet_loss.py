from functools import partial
from typing import Dict
from typing import Dict

import torch
from pytorch_toolbelt.losses import SoftBCEWithLogitsLoss, BinaryBiTemperedLogisticLoss, BinaryFocalLoss, BalancedBCEWithLogitsLoss
from torch import nn, Tensor
from torch.nn import functional as F

from .functional import binary_shannon_entropy_loss, objectness_reduced_focal_loss, objectness_mse_loss
from ..constants import (
    CENTERNET_OUTPUT_SIZE,
    CENTERNET_OUTPUT_OFFSET,
    CENTERNET_OUTPUT_OBJECTNESS_MAP,
    CENTERNET_TARGET_SIZE,
    CENTERNET_TARGET_OFFSET,
    CENTERNET_TARGET_FISHING_MAP,
    CENTERNET_OUTPUT_FISHING_MAP,
    CENTERNET_OUTPUT_VESSEL_MAP,
    CENTERNET_TARGET_VESSEL_MAP,
    CENTERNET_TARGET_OBJECTNESS_MAP,
)
from ...constants import IGNORE_LABEL, TARGET_SAMPLE_WEIGHT

__all__ = ["CircleNetLoss"]


class CircleNetLoss(nn.Module):
    samplewise: bool
    regularize_ignored: bool

    def __init__(
        self,
        objectness_loss: str = "rfl",
        classifier_loss: str = "bce",
        size_loss: str = "mse",
        offset_loss: str = "mse",
        regularize_ignored: bool = False,
        regularization_weight: float = 1.0,
        ignore_index: int = IGNORE_LABEL,
    ):
        super().__init__()
        self.objectness_loss = objectness_loss
        self.regularize_ignored = regularize_ignored
        self.regularization_weight = regularization_weight
        self.ignore_index = ignore_index

        if objectness_loss == "rfl":
            self.objectness_loss = partial(objectness_reduced_focal_loss, ignore_index=self.ignore_index, alpha=2, beta=4, eps=0)
        elif objectness_loss == "mse":
            self.objectness_loss = partial(objectness_mse_loss, ignore_index=self.ignore_index)
        else:
            raise KeyError(objectness_loss)

        if classifier_loss == "bce":
            self.classifier_loss = SoftBCEWithLogitsLoss(ignore_index=ignore_index, reduction="none")
        elif classifier_loss == "bbce":
            self.classifier_loss = BalancedBCEWithLogitsLoss(ignore_index=ignore_index, reduction="none")
        elif classifier_loss == "soft_bce":
            self.classifier_loss = SoftBCEWithLogitsLoss(ignore_index=ignore_index, reduction="none", smooth_factor=0.05)
        elif classifier_loss == "bitempered":
            self.classifier_loss = BinaryBiTemperedLogisticLoss(t1=0.5, t2=4.0, ignore_index=ignore_index, reduction="none")
        elif classifier_loss == "focal":
            self.classifier_loss = BinaryFocalLoss(ignore_index=ignore_index, alpha=None, reduction="none")
        else:
            raise KeyError(classifier_loss)

        if size_loss == "huber":
            self.size_loss = huber_loss
        elif size_loss == "mse":
            self.size_loss = mse_loss
        elif size_loss == "smooth_l1":
            self.size_loss = smooth_l1_loss
        elif size_loss == "rel_error":
            self.size_loss = relative_length_error
        else:
            raise KeyError(size_loss)

        if offset_loss == "mse":
            self.offset_loss = mse_loss
        elif offset_loss == "l1":
            self.offset_loss = l1_loss
        elif offset_loss == "smooth_l1":
            self.offset_loss = smooth_l1_loss
        elif offset_loss == "huber":
            self.offset_loss = huber_loss
        elif offset_loss is None:
            self.offset_loss = None
        else:
            raise KeyError(offset_loss)

    def forward(self, **kwargs) -> Dict[str, Tensor]:
        gt_objectness: Tensor = kwargs[CENTERNET_TARGET_OBJECTNESS_MAP]
        gt_is_vessel: Tensor = kwargs[CENTERNET_TARGET_VESSEL_MAP]
        gt_is_fishing: Tensor = kwargs[CENTERNET_TARGET_FISHING_MAP]
        gt_size: Tensor = kwargs[CENTERNET_TARGET_SIZE]
        sample_weights = kwargs[TARGET_SAMPLE_WEIGHT].reshape(-1, 1, 1, 1)  # [B,1,1,1]

        pred_objectness: Tensor = kwargs[CENTERNET_OUTPUT_OBJECTNESS_MAP]
        pred_is_vessel: Tensor = kwargs[CENTERNET_OUTPUT_VESSEL_MAP]
        pred_is_fishing: Tensor = kwargs[CENTERNET_OUTPUT_FISHING_MAP]
        pred_size: Tensor = kwargs[CENTERNET_OUTPUT_SIZE]

        peaks_mask = gt_objectness.eq(1)
        true_vessels_mask = gt_is_vessel.eq(1)

        num_objects = peaks_mask.sum(dtype=torch.float32).clamp_min(1.0)
        num_vessels = (true_vessels_mask & peaks_mask).sum(dtype=torch.float32).clamp_min(1.0)

        uncertain_objectness_mask = gt_objectness.eq(self.ignore_index)
        certain_objectness_mask = ~uncertain_objectness_mask

        # Returns unreduced losses
        pos_objectness_loss, neg_objectness_loss = self.objectness_loss(pred_objectness, gt_objectness, pos_mask=peaks_mask)

        # Weight classification loss by objectness (This make sense for target encoding=center).
        # This way pixels closer to the center of the keypoint will have bigger influence than ones on the periphery
        vessel_loss = self.classifier_loss(pred_is_vessel, gt_is_vessel) * gt_objectness * certain_objectness_mask
        fishing_loss = self.classifier_loss(pred_is_fishing, gt_is_fishing) * gt_objectness * certain_objectness_mask

        if self.regularize_ignored:
            uncertain_vessel_mask = gt_is_vessel.eq(self.ignore_index)
            uncertain_fishing_mask = (gt_is_fishing.eq(self.ignore_index) & gt_is_vessel.eq(1)) | uncertain_vessel_mask
            # Plus regularization loss for ignored values in the objectness heatmap
            obj_reg_loss = binary_shannon_entropy_loss(pred_objectness) * uncertain_objectness_mask

            vsl_reg_loss = binary_shannon_entropy_loss(pred_is_vessel) * uncertain_vessel_mask

            fsh_reg_loss = binary_shannon_entropy_loss(pred_is_fishing) * uncertain_fishing_mask

            obj_reg_loss = (
                obj_reg_loss.sum(dtype=torch.float32)
                * self.regularization_weight
                / uncertain_objectness_mask.sum(dtype=torch.float32).clamp_min(1.0)
            )

            vsl_reg_loss = (
                vsl_reg_loss.sum(dtype=torch.float32)
                * self.regularization_weight
                / uncertain_vessel_mask.sum(dtype=torch.float32).clamp_min(1.0)
            )
            fsh_reg_loss = (
                fsh_reg_loss.sum(dtype=torch.float32)
                * self.regularization_weight
                / uncertain_fishing_mask.sum(dtype=torch.float32).clamp_min(1.0)
            )
        else:
            obj_reg_loss = 0
            vsl_reg_loss = 0
            fsh_reg_loss = 0

        valid_size_mask = peaks_mask * (gt_size > 0)  # Size loss only for positive lengths
        num_objects_with_length = valid_size_mask.sum(dtype=torch.float32).clamp_min(1.0)
        size_loss = sample_weights * self.size_loss(pred_size, gt_size, valid_size_mask)

        if self.offset_loss is not None:
            offset_loss = self.offset_loss(kwargs[CENTERNET_OUTPUT_OFFSET], kwargs[CENTERNET_TARGET_OFFSET], peaks_mask).sum(
                dim=1, keepdim=True
            )
        else:
            offset_loss = torch.zeros_like(size_loss)

        objectness_loss = neg_objectness_loss + pos_objectness_loss
        mean_obj_loss = (sample_weights * objectness_loss).sum(dtype=torch.float32) / num_objects + obj_reg_loss
        mean_vessel_loss = (sample_weights * vessel_loss).sum(dtype=torch.float32) / num_objects + vsl_reg_loss
        mean_fishing_loss = (sample_weights * fishing_loss).sum(dtype=torch.float32) / num_vessels + fsh_reg_loss
        mean_siz_loss = (sample_weights * size_loss).sum(dtype=torch.float32) / num_objects_with_length

        losses = {
            "objectness": mean_obj_loss.type_as(pred_objectness),
            "vessel": mean_vessel_loss.type_as(pred_objectness),
            "fishing": mean_fishing_loss.type_as(pred_objectness),
            "size": mean_siz_loss.type_as(pred_objectness),
            "vsl_reg_loss": vsl_reg_loss,
            "fsh_reg_loss": fsh_reg_loss,
            "obj_reg_loss": obj_reg_loss,
            "num_objects": num_objects,
            "num_vessels": num_vessels,
            "num_objects_with_length": num_objects_with_length,
        }
        if self.offset_loss is not None:
            mean_off_loss = offset_loss.sum(dtype=torch.float32) / num_objects
            losses["offset"] = mean_off_loss.type_as(pred_objectness)
        return losses


def mse_loss(prediction, target, pos_mask):
    loss = F.mse_loss(prediction, target, reduction="none") * pos_mask
    return loss


def l1_loss(prediction, target, pos_mask):
    loss = F.l1_loss(prediction, target, reduction="none") * pos_mask
    return loss


def smooth_l1_loss(prediction, target, pos_mask):
    loss = F.smooth_l1_loss(prediction, target, reduction="none") * pos_mask
    return loss


def huber_loss(prediction, target, pos_mask):
    loss = F.huber_loss(prediction, target, reduction="none", delta=1) * pos_mask
    return loss


def relative_length_error(prediction, target, pos_mask, eps=1e-4):
    loss = pos_mask * F.smooth_l1_loss(prediction, target, reduction="none") / (target + eps)
    return loss
