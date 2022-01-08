import abc
import os
from abc import abstractmethod
from functools import partial
from typing import Tuple

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


__all__ = [
    "DatasetSplitter",
]


class DatasetSplitter(abc.ABC):
    def __init__(self, data_dir: str, dataset_size):
        self.data_dir = data_dir
        self.dataset_size = dataset_size

    def append_prefix(self, x, folder):
        return os.path.join(self.data_dir, self.dataset_size, folder, x)

    @abstractmethod
    def train_test_split(self, fold: int, num_folds: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        pass
