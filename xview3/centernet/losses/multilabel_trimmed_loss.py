from functools import partial
from typing import Dict, Tuple

import torch
from pytorch_toolbelt.losses import SoftBCEWithLogitsLoss, BinaryBiTemperedLogisticLoss, BinaryFocalLoss, BalancedBCEWithLogitsLoss
from torch import nn, Tensor
from torch.nn import functional as F

from .functional import binary_shannon_entropy_loss
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
from ...constants import IGNORE_LABEL

__all__ = ["MultilabelCenterTrimmedLoss"]


class MultilabelCenterTrimmedLoss(nn.Module):
    fraction: int
    samplewise: bool
    regularize_ignored: bool

    def __init__(
        self,
        fraction: int,
        samplewise: bool = True,
        objectness_loss: str = "rfl",
        classifier_loss="bce",
        size_loss: str = "mse",
        offset_loss: str = "mse",
        regularize_ignored: bool = False,
        regularization_weight: float = 1.0,
        ignore_index: int = IGNORE_LABEL,
    ):
        super().__init__()
        self.fraction = fraction
        self.objectness_loss = objectness_loss
        self.samplewise = samplewise
        self.regularize_ignored = regularize_ignored
        self.regularization_weight = regularization_weight
        self.ignore_index = ignore_index

        if objectness_loss == "rfl":
            self.objectness_loss = partial(self.reduced_focal_loss, alpha=2, beta=4, eps=0)
        elif objectness_loss == "l1":
            self.objectness_loss = self.objectness_l1_loss
        elif objectness_loss == "mse":
            self.objectness_loss = self.objectness_mse_loss
        elif objectness_loss == "focal_mse":
            self.objectness_loss = self.objectness_focal_mse_loss
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
            self.size_loss = self.huber_loss
        elif size_loss == "mse":
            self.size_loss = self.mse_loss
        elif size_loss == "smooth_l1":
            self.size_loss = self.smooth_l1_loss
        elif size_loss == "rel_error":
            self.size_loss = self.relative_length_error
        else:
            raise KeyError(size_loss)

        if offset_loss == "mse":
            self.offset_loss = self.mse_loss
        elif offset_loss == "l1":
            self.offset_loss = self.l1_loss
        elif offset_loss == "smooth_l1":
            self.offset_loss = self.smooth_l1_loss
        elif offset_loss == "huber":
            self.offset_loss = self.huber_loss
        elif offset_loss is None:
            self.offset_loss = None
        else:
            raise KeyError(offset_loss)

    def objectness_mse_loss(self, pred: Tensor, gt: Tensor, pos_mask) -> Tuple[Tensor, Tensor]:
        ign_mask = gt.eq(self.ignore_index)
        pt = pred.sigmoid()
        neg_mask = ~pos_mask
        loss = F.mse_loss(pt, gt, reduction="none")
        pos_loss: Tensor = loss * pos_mask
        neg_loss: Tensor = loss * neg_mask
        return torch.masked_fill(pos_loss, ign_mask, 0), torch.masked_fill(neg_loss, ign_mask, 0)

    def objectness_focal_mse_loss(self, pred: Tensor, gt: Tensor, pos_mask: Tensor, alpha=2.0, beta=2.0) -> Tuple[Tensor, Tensor]:
        pt = pred.sigmoid()
        neg_mask = ~pos_mask
        loss = F.mse_loss(pt, gt, reduction="none")
        pos_loss: Tensor = loss * pos_mask
        neg_loss: Tensor = loss * neg_mask * torch.pow(pt, alpha) * torch.pow(1.0 - gt, beta)
        return pos_loss, neg_loss

    def objectness_l1_loss(self, pred: Tensor, gt: Tensor, pos_mask):
        pt = pred.sigmoid()
        ign_mask = gt.eq(self.ignore_index)
        neg_mask = ~pos_mask
        loss = F.l1_loss(pt, gt, reduction="none")
        pos_loss: Tensor = loss * pos_mask
        neg_loss: Tensor = (loss * neg_mask) ** 4
        return torch.masked_fill(pos_loss, ign_mask, 0), torch.masked_fill(neg_loss, ign_mask, 0)

    def reduced_focal_loss(self, pred: Tensor, gt: Tensor, pos_mask, alpha: float, beta: float, eps: float = 1e-4):
        neg_mask = ~pos_mask
        ign_mask = gt.eq(self.ignore_index)

        pt = pred.sigmoid().clamp(eps, 1 - eps)

        pos_loss: Tensor = -torch.pow(1 - pt, alpha) * F.logsigmoid(pred) * pos_mask
        neg_loss: Tensor = -torch.pow(1.0 - gt, beta) * torch.pow(pt, alpha) * F.logsigmoid(-pred) * neg_mask

        return torch.masked_fill(pos_loss, ign_mask, 0), torch.masked_fill(neg_loss, ign_mask, 0)

    def mse_loss(self, prediction, target, pos_mask):
        loss = F.mse_loss(prediction, target, reduction="none") * pos_mask
        return loss

    def l1_loss(self, prediction, target, pos_mask):
        loss = F.l1_loss(prediction, target, reduction="none") * pos_mask
        return loss

    def smooth_l1_loss(self, prediction, target, pos_mask):
        loss = F.smooth_l1_loss(prediction, target, reduction="none") * pos_mask
        return loss

    def huber_loss(self, prediction, target, pos_mask):
        loss = F.huber_loss(prediction, target, reduction="none", delta=1) * pos_mask
        return loss

    def relative_length_error(self, prediction, target, pos_mask, eps=1e-4):
        loss = pos_mask * F.smooth_l1_loss(prediction, target, reduction="none") / (target + eps)
        return loss

    # @torch.cuda.amp.autocast(False)
    def forward(self, **kwargs) -> Dict[str, Tensor]:
        gt_objectness: Tensor = kwargs[CENTERNET_TARGET_OBJECTNESS_MAP]
        gt_is_vessel: Tensor = kwargs[CENTERNET_TARGET_VESSEL_MAP]
        gt_is_fishing: Tensor = kwargs[CENTERNET_TARGET_FISHING_MAP]
        gt_size: Tensor = kwargs[CENTERNET_TARGET_SIZE]

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
            uncertain_fishing_mask = gt_is_fishing.eq(self.ignore_index)
            # Plus regularization loss for ignored values in the objectness heatmap
            objectness_reg_loss = binary_shannon_entropy_loss(pred_objectness) * uncertain_objectness_mask

            vsl_reg_loss = binary_shannon_entropy_loss(pred_is_vessel) * certain_objectness_mask * uncertain_vessel_mask * gt_objectness
            fsh_reg_loss = (
                binary_shannon_entropy_loss(pred_is_fishing)
                * certain_objectness_mask
                * uncertain_fishing_mask
                * true_vessels_mask
                * gt_objectness
            )

            pos_objectness_loss += objectness_reg_loss * self.regularization_weight
            vessel_loss += vsl_reg_loss * self.regularization_weight
            fishing_loss += fsh_reg_loss * self.regularization_weight

        valid_size_mask = peaks_mask * (gt_size > 0)  # Size loss only for positive lengths
        num_objects_with_length = valid_size_mask.sum(dtype=torch.float32).clamp_min(1.0)
        size_loss = self.size_loss(pred_size, gt_size, valid_size_mask)

        if self.offset_loss is not None:
            offset_loss = self.offset_loss(kwargs[CENTERNET_OUTPUT_OFFSET], kwargs[CENTERNET_TARGET_OFFSET], peaks_mask).sum(
                dim=1, keepdim=True
            )
        else:
            offset_loss = torch.zeros_like(size_loss)

        if self.fraction > 0:
            if self.samplewise:
                (pos_objectness_loss, neg_objectness_loss, vessel_loss, fishing_loss, offset_loss, size_loss,) = samplewise_trimming(
                    pred_objectness,
                    pred_is_vessel,
                    pred_is_fishing,
                    pos_objectness_loss,
                    neg_objectness_loss,
                    vessel_loss,
                    fishing_loss,
                    offset_loss,
                    size_loss,
                )
            else:
                (pos_objectness_loss, neg_objectness_loss, vessel_loss, fishing_loss, offset_loss, size_loss,) = batchwise_trimming(
                    pred_objectness,
                    pred_is_vessel,
                    pred_is_fishing,
                    pos_objectness_loss,
                    neg_objectness_loss,
                    vessel_loss,
                    fishing_loss,
                    offset_loss,
                    size_loss,
                )

        mean_obj_loss = (neg_objectness_loss.sum(dtype=torch.float32) + pos_objectness_loss.sum(dtype=torch.float32)) / num_objects
        mean_vessel_loss = (vessel_loss.sum(dtype=torch.float32)) / num_objects
        mean_fishing_loss = (fishing_loss.sum(dtype=torch.float32)) / num_vessels
        mean_siz_loss = size_loss.sum(dtype=torch.float32) / num_objects_with_length

        losses = {
            "objectness": mean_obj_loss.type_as(pred_objectness),
            "vessel": mean_vessel_loss.type_as(pred_objectness),
            "fishing": mean_fishing_loss.type_as(pred_objectness),
            "size": mean_siz_loss.type_as(pred_objectness),
        }
        if self.offset_loss is not None:
            mean_off_loss = offset_loss.sum(dtype=torch.float32) / num_objects
            losses["offset"] = mean_off_loss.type_as(pred_objectness)
        return losses


def samplewise_trimming(
    pred_objectness,
    pred_is_vessel,
    pred_is_fishing,
    pos_objectness_loss,
    neg_objectness_loss,
    vessel_loss,
    fishing_loss,
    offset_loss,
    size_loss,
    num_elements_to_ignore,
):
    pred_objectness = pred_objectness.flatten(2)  # [B, 1, N]
    pred_is_vessel = pred_is_vessel.flatten(2)  # [B, 1, N]
    pred_is_fishing = pred_is_fishing.flatten(2)  # [B, 1, N]
    pos_objectness_loss = pos_objectness_loss.flatten(2)  # [B, 1, N]
    neg_objectness_loss = neg_objectness_loss.flatten(2)  # [B, 1, N]
    vessel_loss = vessel_loss.flatten(2)  # [B, 1, N]
    fishing_loss = fishing_loss.flatten(2)  # [B, 1, N]
    offset_loss = offset_loss.flatten(2)  # [B, 1, N]
    size_loss = size_loss.flatten(2)  # [B, 1, N]

    # Here we ignore top-N negative losses for negative objectness loss (E.g target is predicted, but groundtruth is negative).
    # Since some objects may be missing in gt, and we don't want to include those wrong false-positives in the training signal
    # And use keep indexes to propagate ignored pixels in other losses
    _, indices = torch.topk(neg_objectness_loss, k=num_elements_to_ignore, dim=2, largest=True, sorted=False)

    non_ignored_mask = torch.ones_like(neg_objectness_loss).scatter_(2, indices, 0)

    # Instead of just setting loss to zeros in those hard example predictions, we minimize entropy loss
    # for objectness & classification to push the predictions to either 0 or 1
    neg_objectness_loss = neg_objectness_loss * non_ignored_mask + binary_shannon_entropy_loss(pred_objectness) * (1 - non_ignored_mask)
    vessel_loss = vessel_loss * non_ignored_mask + binary_shannon_entropy_loss(pred_is_vessel) * (1 - non_ignored_mask)
    fishing_loss = fishing_loss * non_ignored_mask + binary_shannon_entropy_loss(pred_is_fishing) * (1 - non_ignored_mask)
    # For offset & size there is no regularization, so we exclude corresponding pixels from the grad. contribution
    offset_loss = offset_loss * non_ignored_mask
    size_loss = size_loss * non_ignored_mask

    return (
        pos_objectness_loss,
        neg_objectness_loss,
        vessel_loss,
        fishing_loss,
        offset_loss,
        size_loss,
    )


def batchwise_trimming(
    pred_objectness,
    pred_is_vessel,
    pred_is_fishing,
    pos_objectness_loss,
    neg_objectness_loss,
    vessel_loss,
    fishing_loss,
    offset_loss,
    size_loss,
    num_elements_to_ignore,
):
    pred_objectness = pred_objectness.permute(1, 0, 2, 3).flatten(1)  # [1, BxN]
    pred_is_vessel = pred_is_vessel.permute(1, 0, 2, 3).flatten(1)  # [C, BxN]
    pred_is_fishing = pred_is_fishing.permute(1, 0, 2, 3).flatten(1)  # [C, BxN]
    pos_objectness_loss = pos_objectness_loss.permute(1, 0, 2, 3).flatten(1)  # [1, BxN]
    neg_objectness_loss = neg_objectness_loss.permute(1, 0, 2, 3).flatten(1)  # [1, BxN]
    vessel_loss = vessel_loss.permute(1, 0, 2, 3).flatten(1)  # [1, BxN]
    fishing_loss = fishing_loss.permute(1, 0, 2, 3).flatten(1)  # [1, BxN]
    offset_loss = offset_loss.permute(1, 0, 2, 3).flatten(1)  # [1, BxN]
    size_loss = size_loss.permute(1, 0, 2, 3).flatten(1)  # [1, BxN]

    # Here we ignore top-N negative losses for negative objectness loss (E.g target is predicted, but groundtruth is negative).
    # Since some objects may be missing in gt, and we don't want to include those wrong false-positives in the training signal
    # And use keep indexes to propagate ignored pixels in other losses

    _, indices = torch.topk(neg_objectness_loss, k=num_elements_to_ignore, dim=1, largest=True, sorted=False)

    non_ignored_mask = torch.ones_like(neg_objectness_loss).scatter_(1, indices, 0)

    # Instead of just setting loss to zeros in those hard example predictions, we minimize entropy loss
    # for objectness & classification to push the predictions to either 0 or 1
    neg_objectness_loss = neg_objectness_loss * non_ignored_mask + binary_shannon_entropy_loss(pred_objectness) * (1 - non_ignored_mask)
    vessel_loss = vessel_loss * non_ignored_mask + binary_shannon_entropy_loss(pred_is_vessel) * (1 - non_ignored_mask)
    fishing_loss = fishing_loss * non_ignored_mask + binary_shannon_entropy_loss(pred_is_fishing) * (1 - non_ignored_mask)
    # For offset & size there is no regularization, so we exclude corresponding pixels from the grad. contribution
    offset_loss = offset_loss * non_ignored_mask
    size_loss = size_loss * non_ignored_mask

    return (
        pos_objectness_loss,
        neg_objectness_loss,
        vessel_loss,
        fishing_loss,
        offset_loss,
        size_loss,
    )
