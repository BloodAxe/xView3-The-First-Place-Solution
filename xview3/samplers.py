import collections
import math
import random
from typing import Iterator, Union, Optional
from typing import List

import numpy as np
from catalyst.contrib.utils.misc import find_value_ids
from sklearn.utils import compute_sample_weight
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, WeightedRandomSampler

from .dataset import KeypointsDataset

__all__ = [
    "ClassBalancedDataset",
    "RepeatedBalanceBatchSampler",
    "labels_to_class_weights",
    "labels_to_image_weights",
    "compute_balancing_weights",
]


class LocationBalancedSampler(WeightedRandomSampler):
    def __init__(self, dataset: KeypointsDataset, num_samples: int):
        locations = [fs.id_from_fname(x)[:3] for x in dataset.masks]
        weights = compute_sample_weight("balanced", locations)
        super().__init__(weights, num_samples)


class FloodingBalancedSampler(WeightedRandomSampler):
    def __init__(self, dataset: KeypointsDataset, num_samples: int):
        flooding_present = [np.any(read_mask_image(x)[0] == 1) for x in dataset.masks]
        weights = compute_sample_weight("balanced", flooding_present)
        super().__init__(weights, num_samples)


class LocationAndFloodingBalancedSampler(WeightedRandomSampler):
    def __init__(self, dataset: KeypointsDataset, num_samples: int):
        locations = [fs.id_from_fname(x)[:3] for x in dataset.masks]
        loc_weights = compute_sample_weight("balanced", locations)

        flooding_present = [np.any(read_mask_image(x)[0] == 1) for x in dataset.masks]
        pos_weights = compute_sample_weight("balanced", flooding_present)

        weights = np.sqrt(loc_weights * pos_weights)
        super().__init__(weights, num_samples)


class ClassBalancedDataset(Dataset):
    """A wrapper of repeated dataset with repeat factor.
    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, an image may appear multiple times based on its
    "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :func:`self.get_cat_ids` to support
    ClassBalancedDataset.
    The repeat factor is computed as followed.
    1. For each category c, compute the fraction # of images
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor:
       :math:`r(I) = max_{c in I} r(c)`
    Args:
        dataset (:obj:`CustomDataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes will not be oversampled. Otherwise, they will be categorized
            as the pure background class and involved into the oversampling.
            Default: True.
    """

    def __init__(
        self,
        dataset,
        oversample_thr: float,
        filter_empty_gt=True,
    ):
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.filter_empty_gt = filter_empty_gt
        self.classes = dataset.classes

        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, "flag"):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset, repeat_thr):
        """Get repeat factor for each images in the dataset.
        Args:
            dataset (:obj:`CustomDataset`): The dataset
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.
        Returns:
            list[float]: The repeat factors for each images in the dataset.
        """

        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = collections.defaultdict(int)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.classes)])
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq)) for cat_id, cat_freq in category_freq.items()}

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.classes)])
            repeat_factor = 1
            if len(cat_ids) > 0:
                repeat_factor = max({category_repeat[cat_id] for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def __len__(self):
        """Length after repetition."""
        return len(self.repeat_indices)

    def get_collate_fn(self):
        return self.dataset.get_collate_fn()


class RepeatedBalanceBatchSampler(Sampler):
    """
    This kind of sampler can be used for both metric learning and
    classification task.

    Sampler with the given strategy for the C unique classes dataset:
    - Selection P of C classes for the 1st batch
    - Selection K instances for each class for the 1st batch
    - Selection P of C - P remaining classes for 2nd batch
    - Selection K instances for each class for the 2nd batch
    - ...
    The epoch ends when there are no classes left.
    So, the batch sise is P * K except the last one.

    Thus, in each epoch, all the classes will be selected once, but this
    does not mean that all the instances will be selected during the epoch.

    One of the purposes of this sampler is to be used for
    forming triplets and pos/neg pairs inside the batch.
    To guarante existance of these pairs in the batch,
    P and K should be > 1. (1)

    Behavior in corner cases:
    - If a class does not contain K instances,
    a choice will be made with repetition.
    - If C % P == 1 then one of the classes should be dropped
    otherwise statement (1) will not be met.

    This type of sampling can be found in the classical paper of Person Re-Id,
    where P equals 32 and K equals 4:
    `In Defense of the Triplet Loss for Person Re-Identification`_.

    Args:
        labels: list of classes labeles for each elem in the dataset
        num_classes: number of classes in a batch, should be > 1
        k: number of instances of each class in a batch, should be > 1

    .. _In Defense of the Triplet Loss for Person Re-Identification:
        https://arxiv.org/abs/1703.07737
    """

    def __init__(self, labels: Union[List[int], np.ndarray], num_classes_in_batch: int, batch_size: int, repeats: int):
        """Sampler initialisation."""
        super().__init__(self)

        classes = set(labels)
        num_samples_of_single_class = batch_size // num_classes_in_batch
        if batch_size % num_classes_in_batch != 0:
            raise ValueError(f"")
        counter = collections.Counter(labels)

        assert all(n > 1 for n in counter.values()), "Each class should contain at least 2 instances to fit (1)"
        assert (1 < num_classes_in_batch <= len(classes)) and (1 < num_samples_of_single_class)

        self._labels = labels
        self.num_classes_in_batch = num_classes_in_batch
        self.num_samples_of_single_class = num_samples_of_single_class

        self._batch_size = num_classes_in_batch * num_samples_of_single_class
        self._classes = classes
        self._repeats = repeats

        # to satisfy statement (1)
        num_classes = len(self._classes)
        self._num_epoch_classes = (num_classes // num_classes_in_batch) * num_classes_in_batch

    @property
    def batch_size(self) -> int:
        """
        Returns:
            this value should be used in DataLoader as batch size
        """
        return self._batch_size

    # @property
    # def batches_in_epoch(self) -> int:
    #     """
    #     Returns:
    #         number of batches in an epoch
    #     """
    #     return int(np.ceil(self._num_epoch_classes / self._p)) * self._repeats

    def __len__(self) -> int:
        """
        Returns:
            number of samples in an epoch
        """
        return self._num_epoch_classes * self.num_samples_of_single_class * self._repeats

    def __iter__(self) -> Iterator[int]:
        """
        Returns:
            indeces for sampling dataset elems during an epoch
        """
        inds = []

        for _ in range(self._repeats):
            for cls_id in random.sample(self._classes, self._num_epoch_classes):
                all_cls_inds = find_value_ids(self._labels, cls_id)

                # we've checked in __init__ that this value must be > 1
                num_samples_exists = len(all_cls_inds)

                if num_samples_exists < self.num_samples_of_single_class:
                    selected_inds = random.sample(all_cls_inds, k=num_samples_exists) + random.choices(
                        all_cls_inds, k=self.num_samples_of_single_class - num_samples_exists
                    )
                else:
                    selected_inds = random.sample(all_cls_inds, k=self.num_samples_of_single_class)

                inds.extend(selected_inds)

        return iter(inds)


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
