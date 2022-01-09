from typing import List
from typing import Optional

import numpy as np
from sklearn.utils import compute_sample_weight

__all__ = [
    "labels_to_class_weights",
    "labels_to_image_weights",
    "compute_balancing_weights",
]


def labels_to_class_weights(labels: List, num_classes: int):
    # Get class weights (inverse frequency) from training labels

    weights = np.bincount(labels, minlength=num_classes)  # occurences per class

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return weights


def labels_to_image_weights(labels: List, num_classes: int, class_weights: np.ndarray):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array([np.bincount(labels[i], minlength=num_classes) for i in range(n)])
    image_weights = (class_weights.reshape(1, num_classes) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def compute_balancing_weights(
    labels: np.ndarray,
    num_samples: int,
    min_samples_per_class: Optional[int],
    max_samples_per_class: Optional[int],
    sqrt_weights: bool = False,
) -> np.ndarray:
    weights = compute_sample_weight("balanced", labels)
    if sqrt_weights:
        weights = np.sqrt(weights)
    classes = np.unique(labels)

    for class_label in classes:
        mask = labels == class_label
        num_samples_of_class = mask.sum()
        estimated_samples = weights[mask].sum() * num_samples / len(labels)
        if min_samples_per_class and (estimated_samples < min_samples_per_class):
            adjusted_weight = ((min_samples_per_class * len(labels)) / num_samples) / num_samples_of_class
            weights[mask] = adjusted_weight

        if max_samples_per_class and (estimated_samples > max_samples_per_class):
            adjusted_weight = ((max_samples_per_class * len(labels)) / num_samples) / num_samples_of_class
            weights[mask] = adjusted_weight

    return weights
