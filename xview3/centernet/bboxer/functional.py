import math
from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "bes_radius",
    "centernet_circle_2d",
    "centernet_gather_feat",
    "centernet_gaussian_2d",
    "centernet_heatmap_nms",
    "centernet_topk",
    "centernet_tranpose_and_gather_feat",
    "draw_umich_gaussian",
    "draw_msra_gaussian",
    "gaussian_radius",
    "pointwise_gaussian_2d",
    "centernet_tight_heatmap_nms",
    "compute_phi",
    "one_hot_labels",
    "change_box_order",
    "box_clamp",
    "box_iou",
    "box_select",
    "meshgrid",
    "bboxes_area_xyxy",
]


def bboxes_area_xyxy(bboxes: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """

    Args:
        bboxes: Tensor of bboxes [N,4] in xyxy format

    Returns:

    """
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    return w * h


def centernet_gaussian_2d(shape, sigma=1.0):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def centernet_circle_2d(r: int, value: float, dtype):
    a, b = r, r
    n = 2 * r + 1

    y, x = np.ogrid[-a : n - a, -b : n - b]
    mask = x * x + y * y <= r * r

    array = np.zeros((n, n), dtype=dtype)
    value = np.array(value, dtype=dtype).reshape((1, 1))
    array[:] = value
    array[~mask] = 0
    return array


def pointwise_gaussian_2d():
    pos_kernel = np.float32([[0.5, 0.75, 0.5], [0.75, 1.0, 0.75], [0.5, 0.75, 0.5]])
    return pos_kernel


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    r = min(r1, r2, r3)
    return max(1, math.ceil(r))


def compute_phi(w, h):
    a, b = min(w, h), max(w, h)
    aspect_ratio = a / b
    angle = 45 * aspect_ratio
    if w > h:
        angle = 90 - angle
    return angle


def bes_radius(det_size, min_overlap=0.5) -> int:
    """Compute radius of gaussian.
    Arguments:
        w (int): weight of box.
        h (int): height of box.
        iou_threshold (float): min required IOU between gt and smoothed box.
    Returns:
        radius (int): radius of gaussian.
    """
    w, h = det_size
    phi = compute_phi(w, h)
    sin_phi = math.sin(math.radians(phi))
    cos_phi = math.cos(math.radians(phi))
    a = sin_phi * cos_phi
    b = -(w * sin_phi + h * cos_phi)
    c = w * h * (1 - min_overlap) / (1 + min_overlap)
    d = math.sqrt(b * b - 4 * a * c)
    r = -(b + d) / (2 * a)
    return int(max(1, math.ceil(r)))


def gaussian2DNonSquare(size_x, size_y, alpha, dtype=np.float32):
    y, x = np.ogrid[0:size_y, 0:size_x]
    mx = size_x // 2
    my = size_y // 2
    sigma_x = alpha * size_x / 6
    sigma_y = alpha * size_y / 6
    h = np.exp(-((x - mx) ** 2.0 / (2.0 * sigma_x ** 2.0) + (y - my) ** 2.0 / (2.0 * sigma_y ** 2.0))).astype(dtype)
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    area = np.sum(h > 0)
    return h, area


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.maximum(
        heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]], g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    )
    return heatmap


def draw_umich_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
    if radius == "pointwise":
        gaussian = pointwise_gaussian_2d()
        radius = 1
    else:
        diameter = 2 * radius + 1
        gaussian = centernet_gaussian_2d((diameter, diameter), sigma=diameter / 6.0)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap


def draw_circle(size_map: np.ndarray, center: Tuple[int, int], radius: int, value: float):
    """
    Draw bbox size or offset in a radius
    :param size_map: [2, rows, cols]
    :param center: (x,y)
    :param radius: int
    :param value: [2]
    :return:
    """
    gaussian = centernet_circle_2d(radius, value, dtype=size_map.dtype)

    x, y = int(center[0]), int(center[1])

    height, width = size_map.shape

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = size_map[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return size_map


def disk3_max_pool2d(heatmap):
    hp = torch.max_pool2d(heatmap, kernel_size=(3, 1), padding=(1, 0), stride=1)
    vp = torch.max_pool2d(heatmap, kernel_size=(1, 3), padding=(0, 1), stride=1)
    keep = (heatmap == hp) & (heatmap == vp)
    return heatmap * keep.type_as(heatmap)


def disk5_max_pool2d(heatmap):
    hp = torch.max_pool2d(heatmap, kernel_size=(5, 1), padding=(2, 0), stride=1)
    vp = torch.max_pool2d(heatmap, kernel_size=(1, 5), padding=(0, 2), stride=1)
    mp = torch.max_pool2d(heatmap, kernel_size=(3, 3), padding=(1, 1), stride=1)
    keep = (heatmap == hp) & (heatmap == vp) & (heatmap == mp)
    return heatmap * keep.type_as(heatmap)


def centernet_heatmap_nms(heatmap: Tensor, kernel=3) -> Tensor:
    pad = (kernel - 1) // 2

    hmax = torch.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).type_as(heatmap)
    return heatmap * keep


def centernet_tight_heatmap_nms(heatmap: Tensor, kernel=3) -> Tensor:
    if kernel == 3:
        return disk3_max_pool2d(heatmap)
    elif kernel == 5:
        return disk5_max_pool2d(heatmap)
    else:
        raise KeyError(f"Unsupported kernel size {kernel}")


def centernet_topk(scores: Tensor, top_k: int = 100) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), top_k)

    topk_inds = topk_inds % (height * width)
    topk_ys = torch.div(topk_inds, width, rounding_mode="trunc").int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), top_k)
    topk_clses = torch.div(topk_ind, top_k, rounding_mode="trunc").int()
    topk_inds = centernet_gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, top_k)
    topk_ys = centernet_gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, top_k)
    topk_xs = centernet_gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, top_k)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def centernet_gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def centernet_tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = centernet_gather_feat(feat, ind)
    return feat


def one_hot_labels(tensor, num_classes: int):
    """Encodes class labes as one-hot vector. Negative labels converted too background class.

    :param tensor: Labels vector of shape [N]
    :param num_classes: Number of classes, including dummy background class. E.g num_real_classes + 1
    :return: Labels tensor of shape [N, num_classes]
    """
    if len(tensor) == 0:
        return torch.from_numpy(np.array([], dtype=int)).long()

    tensor = tensor.clone()
    tensor[tensor < 0] = 0
    one_hot_tensor = torch.zeros(tensor.size() + torch.Size([num_classes]))
    dim = len(tensor.size())
    index = tensor.unsqueeze(dim)
    return one_hot_tensor.scatter_(dim, index, 1)


def change_box_order(boxes, order, dim=-1):
    """Change box order between (x_min, y_min, x_max, y_max) and (x_center, y_center, width, height).

    Args:
      boxes: (tensor) bounding boxes, sized [N, 4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N, 4].
    """
    assert order in ["xyxy2xywh", "xywh2xyxy"]
    a = boxes[..., 0:2]
    b = boxes[..., 2:4]
    if order == "xyxy2xywh":
        return torch.cat([(a + b) / 2, b - a], dim)
    return torch.cat([a - b / 2, a + b / 2], dim)


def box_clamp(boxes, x_min, y_min, x_max, y_max):
    """Clamp boxes.

    Args:
      boxes: (tensor) bounding boxes of (x_min, y_min, x_max, y_max), sized [N, 4].
      x_min: (number) min value of x.
      y_min: (number) min value of y.
      x_max: (number) max value of x.
      y_max: (number) max value of y.

    Returns:
      (tensor) clamped boxes.
    """
    boxes[:, 0] = boxes[:, 0].clamp(min=x_min, max=x_max)
    boxes[:, 1] = boxes[:, 1].clamp(min=y_min, max=y_max)
    boxes[:, 2] = boxes[:, 2].clamp(min=x_min, max=x_max)
    boxes[:, 3] = boxes[:, 3].clamp(min=y_min, max=y_max)
    return boxes


def box_select(boxes, x_min, y_min, x_max, y_max):
    """Select boxes in range (x_min, y_min, x_max, y_max).

    Args:
      boxes: (tensor) bounding boxes of (x_min, y_min, x_max, y_max), sized [N, 4].
      x_min: (number) min value of x.
      y_min: (number) min value of y.
      x_max: (number) max value of x.
      y_max: (number) max value of y.

    Returns:
      (tensor) selected boxes, sized [M, 4].
      (tensor) selected mask, sized [N, ].
    """
    mask = (boxes[:, 0] >= x_min) & (boxes[:, 1] >= y_min) & (boxes[:, 2] <= x_max) & (boxes[:, 3] <= y_max)
    boxes = boxes[mask, :]
    return boxes, mask


def box_iou(box1, box2):
    """Compute the intersection over union of two set of boxes.

    The box order must be (x_min, y_min, x_max, y_max).

    Args:
      box1: (tensor) bounding boxes, sized [N, 4].
      box2: (tensor) bounding boxes, sized [M, 4].

    Return:
      (tensor) iou, sized [N, M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    # N = box1.size(0)
    # M = box2.size(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N, ]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M, ]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def meshgrid(x, y, row_major=True):
    """Return meshgrid in range x & y.

    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.

    Returns:
      (tensor) meshgrid, sized [x * y, 2]

    Example:
    >> meshgrid(3, 2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]

    >> meshgrid(3, 2, row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    """
    a = torch.arange(0, x, dtype=torch.float)
    b = torch.arange(0, y, dtype=torch.float)
    xx = a.repeat(y).view(-1, 1)
    yy = b.view(-1, 1).repeat(1, x).view(-1, 1)
    return torch.cat([xx, yy], 1) if row_major else torch.cat([yy, xx], 1)
