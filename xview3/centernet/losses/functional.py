from typing import Tuple

import torch
import torch.nn.functional as F
from pytorch_toolbelt.losses.functional import focal_loss_with_logits, softmax_focal_loss_with_logits
from torch import Tensor

__all__ = [
    "objectness_mse_loss",
    "objectness_reduced_focal_loss",
    "reduced_focal_loss_per_image",
    "reduced_focal_loss",
    "squeeze_heatmap",
    "shannon_entropy_loss",
    "binary_shannon_entropy_loss",
]


def objectness_mse_loss(pred: Tensor, gt: Tensor, pos_mask, ignore_index) -> Tuple[Tensor, Tensor]:
    neg_mask = ~pos_mask
    ign_mask = gt.eq(ignore_index)

    pt = pred.sigmoid()

    loss = F.mse_loss(pt, gt, reduction="none")
    pos_loss: Tensor = loss * pos_mask
    neg_loss: Tensor = loss * neg_mask
    return torch.masked_fill(pos_loss, ign_mask, 0), torch.masked_fill(neg_loss, ign_mask, 0)


def objectness_reduced_focal_loss(pred: Tensor, gt: Tensor, pos_mask, ignore_index, alpha: float, beta: float, eps: float = 1e-4):
    neg_mask = ~pos_mask
    ign_mask = gt.eq(ignore_index)

    pt = pred.sigmoid().clamp(eps, 1 - eps)

    pos_loss: Tensor = -torch.pow(1 - pt, alpha) * F.logsigmoid(pred) * pos_mask
    neg_loss: Tensor = -torch.pow(1.0 - gt, beta) * torch.pow(pt, alpha) * F.logsigmoid(-pred) * neg_mask

    return torch.masked_fill(pos_loss, ign_mask, 0), torch.masked_fill(neg_loss, ign_mask, 0)


def squeeze_heatmap(heatmap: Tensor, keepdim: bool = True) -> Tensor:
    """

    Args:
        heatmap: [B,C,H,W]
        keepdim:

    Returns:
        Binary tensor of shape [B, 1, H, W], where True values corresponds to peaks in the heatmap
    """
    pos_mask = heatmap.eq(1).any(dim=1, keepdim=keepdim)
    return pos_mask


def reduced_focal_loss(pred: Tensor, gt: Tensor, alpha: float, beta: float, eps: float = 1e-4) -> Tensor:
    pos_inds = gt.eq(1.0)
    num_pos = pos_inds.sum()

    pos_mask = pos_inds
    neg_mask = ~pos_mask

    pt = pred.sigmoid().clamp(eps, 1 - eps)

    pos_loss: Tensor = torch.pow(1 - pt, alpha) * torch.nn.functional.logsigmoid(pred) * pos_mask
    neg_loss: Tensor = torch.pow(1.0 - gt, beta) * torch.pow(pt, alpha) * torch.nn.functional.logsigmoid(-pred) * neg_mask

    pos_loss = -pos_loss.sum(dtype=torch.float32)
    neg_loss = -neg_loss.sum(dtype=torch.float32)
    loss = (neg_loss + pos_loss) / num_pos.clamp_min(1)
    return loss


def reduced_focal_loss_per_image(pred: Tensor, gt: Tensor, alpha: float, beta: float, eps: float = 1e-4) -> Tensor:
    pos_inds = gt.eq(1.0)
    num_pos = pos_inds.sum(dtype=torch.float32, dim=(1, 2, 3))  # [B]

    pos_mask = pos_inds
    neg_mask = ~pos_mask

    pt = pred.sigmoid().clamp(eps, 1 - eps)

    pos_loss: Tensor = torch.pow(1 - pt, alpha) * torch.nn.functional.logsigmoid(pred) * pos_mask
    neg_loss: Tensor = torch.pow(1.0 - gt, beta) * torch.pow(pt, alpha) * torch.nn.functional.logsigmoid(-pred) * neg_mask

    pos_loss = -pos_loss.sum(dtype=torch.float32, dim=(1, 2, 3))
    neg_loss = -neg_loss.sum(dtype=torch.float32, dim=(1, 2, 3))
    loss = (neg_loss + pos_loss) / num_pos.clamp_min(1)
    return loss.mean()


def balanced_binary_cross_entropy_with_logits(outputs: Tensor, targets: Tensor, reduction: str = "mean", gamma=1.0) -> Tensor:
    targets = targets.float()
    outputs = outputs.float()

    one_minus_beta: Tensor = targets.mean()
    beta = 1.0 - one_minus_beta

    # one_minus_beta: Tensor = targets.eq(1).sum(dtype=torch.float32) / targets.numel()
    # beta = 1 - one_minus_beta

    # beta = targets.eq(0).sum(dtype=torch.float32) / float(targets.numel())
    # one_minus_beta: Tensor = 1 - beta

    # beta = targets.lt(0.5).sum(dtype=outputs.dtype) / targets.numel()
    # one_minus_beta: Tensor = 1 - beta

    pos_term = beta * targets.pow(gamma) * torch.nn.functional.logsigmoid(outputs)
    neg_term = one_minus_beta * (1 - targets).pow(gamma) * torch.nn.functional.logsigmoid(-outputs)

    loss = -(pos_term + neg_term)

    if reduction == "mean":
        loss = loss.mean()

    if reduction == "sum":
        loss = loss.sum()

    return loss


def centernet_classification_bce_loss(prediction: Tensor, target_classes: Tensor) -> Tensor:
    # If there is no label, we ignore this pixel
    ignore_mask = target_classes.sum(dim=1, keepdim=True).eq(0)

    loss = F.binary_cross_entropy_with_logits(prediction, target_classes, reduction="none")
    keep_mask = (~ignore_mask).type_as(loss)
    num_pos = keep_mask.sum(dtype=torch.float32).clamp_min(1)
    loss = (loss * keep_mask).sum(dtype=torch.float32)
    return loss / num_pos


def centernet_classification_ce_loss(prediction: Tensor, target_classes: Tensor, weight_mask: Tensor) -> Tensor:
    # If two classes overlap or there is no label, we ignore this pixel
    target_labels = target_classes.argmax(dim=1)

    num_pos = weight_mask.sum().clamp_min(1)
    loss = F.cross_entropy(prediction, target_labels, reduction="none") * weight_mask
    return loss.sum() / num_pos


def cnet_cls_focal_heatmap(prediction: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    pos_mask = squeeze_heatmap(target)
    num_pos = pos_mask.sum().clamp_min(eps)

    loss = focal_loss_with_logits(prediction, target, alpha=None, gamma=2, reduction="none")
    loss_sum = (loss * pos_mask).sum()
    loss_mean = loss_sum / num_pos
    return loss_mean


def cnet_cls_softmax_focal_heatmap(prediction: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    target_labels = torch.argmax(target, dim=1)

    pos_mask = squeeze_heatmap(target)
    num_pos = pos_mask.sum().clamp_min(eps)

    loss = softmax_focal_loss_with_logits(prediction, target_labels, gamma=2, reduction="none")
    loss_sum = (loss * pos_mask).sum()
    loss_mean = loss_sum / num_pos
    return loss_mean


def shannon_entropy_loss(x, dim=1):
    sm = x.softmax(dim=dim)
    lsm = F.log_softmax(x, dim=dim)
    entropy = (sm * lsm).sum(dim=dim, keepdim=True)
    return -entropy


def binary_shannon_entropy_loss(x):
    pt = x.sigmoid()
    entropy = F.logsigmoid(x) * pt + F.logsigmoid(-x) * (1 - pt)
    return -entropy
