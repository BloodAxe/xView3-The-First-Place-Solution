import sys
import traceback
import warnings

import catalyst
import numpy as np
import pandas as pd
import torch

from xview3.early_stopping import EarlyStoppingCallback


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback

from functools import partial
from typing import Optional, Tuple, List

from catalyst.callbacks import ControlFlowCallback
from catalyst.core import Callback
from omegaconf import DictConfig
from pytorch_toolbelt.utils.catalyst import (
    ShowPolarBatchesCallback,
    BestMetricCheckpointCallback,
    HyperParametersCallback,
)
from torch import nn
from torch.utils.data import Dataset, Sampler

from xview3 import *
from xview3.centernet import *


class CenternetPipeline(Pipeline):
    valid_df: Optional[pd.DataFrame] = None

    def __init__(self, cfg):
        super().__init__(cfg)
        self.box_coder: Optional[MultilabelCircleNetCoder] = None
        self.shore_root = None

    def build_datasets(self, config) -> Tuple[Dataset, Dataset, Optional[Sampler], List[Callback]]:
        dataset = MultilabelCircleNetDataModule(data_dir=config.dataset.data_dir)
        train_df, valid_df, _, shore_root = dataset.train_val_split(
            splitter=config.dataset.splitter,
            fold=config.dataset.fold,
            num_folds=config.dataset.num_folds,
        )

        self.valid_df = valid_df.copy()  # Keep a copy of unmodified groundtruth

        self.master_print("Dataset", config.dataset.slug)
        self.master_print("\tignore_low_confidence_detections", config.dataset.ignore_low_confidence_detections)
        self.master_print("\tignore_low_confidence_labels    ", config.dataset.ignore_low_confidence_labels)
        self.master_print("\tfilter_low_confidence_objects   ", config.dataset.filter_low_confidence_objects)

        self.master_print("Train dataset")
        self.master_print("\tUnknown vessels", np.isnan(train_df.is_vessel.values.astype(float)).sum())
        self.master_print("\tUnknown fishing", np.isnan(train_df.is_fishing.values.astype(float)).sum())
        self.master_print("Valid dataset")
        self.master_print("\tUnknown vessels", np.isnan(valid_df.is_vessel.values.astype(float)).sum())
        self.master_print("\tUnknown fishing", np.isnan(valid_df.is_fishing.values.astype(float)).sum())

        if config.dataset.filter_low_confidence_objects and config.dataset.ignore_low_confidence_labels:
            raise ValueError(f"filter_low_confidence_objects and ignore_low_confidence_labels are mutually exclusive")

        if config.dataset.filter_low_confidence_objects:
            train_df = filter_low_confidence_objects(train_df)
            valid_df = filter_low_confidence_objects(valid_df)
            self.master_print("Excluding low-confidence objects from training & validation")
        elif config.dataset.ignore_low_confidence_labels:
            train_df = ignore_low_confidence_objects(train_df)
            valid_df = ignore_low_confidence_objects(valid_df)
            self.master_print("Ignoring low-confidence labels from training & validation")

        self.shore_root = shore_root

        import albumentations as A

        augmentations = build_augmentations(config.augs.spatial)
        normalization = build_normalization(config.normalization)

        individual_augmentations = dict(
            [(key, A.Compose(build_augmentations(value))) for key, value in config.augs.individual.items() if value is not None]
        )

        if config.sampler.sampler_type == "each_object":
            train_ds, train_sampler = dataset.get_random_crop_each_object_dataset(
                train_df,
                train_image_size=config.train.train_image_size,
                input_channels=config.dataset.channels,
                box_coder=self.box_coder,
                num_samples=config.sampler.num_samples,
                individual_augmentations=individual_augmentations,
                augmentations=augmentations,
                normalization=normalization,
                balance_crowd=config.sampler.balance_crowd,
                balance_near_shore=config.sampler.balance_near_shore,
                balance_vessel_type=config.sampler.balance_vessel_type,
                balance_fishing_type=config.sampler.balance_fishing_type,
                balance_per_scene=config.sampler.balance_per_scene,
                balance_location=config.sampler.balance_location,
                channels_last=config.torch.channels_last,
            )
        else:
            train_ds, train_sampler = dataset.get_random_crop_training_dataset(
                train_df,
                train_image_size=config.train.train_image_size,
                input_channels=config.dataset.channels,
                box_coder=self.box_coder,
                num_samples=config.sampler.num_samples,
                crop_around_ship_p=config.sampler.crop_around_ship_p,
                individual_augmentations=individual_augmentations,
                augmentations=augmentations,
                normalization=normalization,
                balance_near_shore=config.sampler.balance_near_shore,
                balance_crowd=config.sampler.balance_crowd,
                balance_location=config.sampler.balance_location,
                balance_folder=config.sampler.balance_folder,
                channels_last=config.torch.channels_last,
                shore_root=shore_root,
            )

        valid_ds = dataset.get_validation_dataset(
            valid_df,
            valid_crop_size=config.train.valid_image_size,
            input_channels=config.dataset.channels,
            box_coder=self.box_coder,
            normalization=normalization,
            channels_last=config.torch.channels_last,
        )

        self.master_print(
            "Train",
            "instances",
            len(train_df),
            "dataset",
            len(train_ds),
            "sampler",
            len(train_sampler) if train_sampler else "N/A",
        )
        self.master_print("Valid", "instances", len(valid_df), "dataset", len(valid_ds))

        return train_ds, valid_ds, train_sampler, []

    def build_metrics(self, config, loaders, model):
        callbacks = [
            CircleNetRocAucMetricsCallback(prefix="extra_metrics"),
            MultilabelCircleNetDecodeCallback(box_coder=self.box_coder),
        ]

        show_batches = self.cfg["train"].get("show", False)

        callbacks += [
            # This is the vanilla method for computing score that matches the public LB atm.
            # ControlFlowCallback(
            #     MultilabelCircleNetDetectionMetrics(
            #         groundtruth_df=filter_low_confidence_objects(self.valid_df),
            #         shore_root=self.shore_root,
            #         prefix="conf_",
            #         objectness_thresholds=[0.5],
            #         is_vessel_threshold=[0.5],
            #         is_fishing_threshold=[0.5],
            #         loc_performance_matching_method=compute_loc_performance,
            #     ),
            #     loaders="valid",
            # ),
            # This is the 'right' method for computing score.
            ControlFlowCallback(
                MultilabelCircleNetDetectionMetrics(
                    groundtruth_df=self.valid_df,
                    shore_root=self.shore_root,
                    prefix="",
                    objectness_thresholds=[0.25, 0.275, 0.3, 0.325, 0.350, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55],
                ),
                loaders="valid",
            ),
        ]

        if self.is_master:
            callbacks += [
                # BestMetricCheckpointCallback(
                #     target_metric="metrics/conf_aggregate",
                #     target_metric_minimize=False,
                #     save_n_best=3,
                # ),
                BestMetricCheckpointCallback(
                    target_metric="metrics/aggregate",
                    target_metric_minimize=False,
                    save_n_best=3,
                ),
            ]

        if config["train"]["early_stopping"] > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    metrics=[
                        # "metrics/conf_aggregate",
                        "metrics/aggregate",
                    ],
                    minimize=False,
                    min_delta=1e-6,
                    patience=config["train"]["early_stopping"],
                )
            )

        if self.is_master:
            if show_batches:
                visualize_fn = visualize_predictions_multilabel_centernet
                visualize_batch = partial(
                    visualize_fn,
                    box_coder=self.box_coder,
                    min_confidence_score=0.3,
                    # max_image_size=2048,
                )
                callbacks.append(ShowPolarBatchesCallback(visualize_batch, metric="loss", minimize=True))

            callbacks.append(
                HyperParametersCallback(
                    hparam_dict=dict(
                        # Model stuff
                        model=str(self.cfg.model.config.slug),
                        model_output_stride=model.box_coder.output_stride,
                        # Box Coder
                        box_coder_heatmap_encoding=str(self.cfg.model.config.box_coder.heatmap_encoding),
                        box_coder_labels_encoding=str(self.cfg.model.config.box_coder.labels_encoding),
                        box_coder_fixed_radius=str(self.cfg.model.config.box_coder.fixed_radius),
                        box_coder_labels_radius=str(self.cfg.model.config.box_coder.labels_radius),
                        # Optimizer
                        optimizer=str(self.cfg.optimizer.name),
                        optimizer_lr=float(self.cfg.optimizer.params.lr),
                        optimizer_eps=float(self.cfg.optimizer.params.eps) if "eps" in self.cfg.optimizer.params else "None",
                        optimizer_wd=float(self.cfg.optimizer.params.weight_decay),
                        optimizer_scheduler=str(self.cfg.scheduler.scheduler_name),
                        # Dataset
                        dataset=str(self.cfg.dataset.slug),
                        dataset_fold=int(self.cfg.dataset.fold) if self.cfg.dataset.fold is not None else "None",
                        dataset_filter_low_confidence_objects=bool(self.cfg.dataset.filter_low_confidence_objects),
                        dataset_ignore_low_confidence_labels=bool(self.cfg.dataset.ignore_low_confidence_labels),
                        dataset_ignore_low_confidence_detections=bool(self.cfg.dataset.ignore_low_confidence_detections),
                        dataset_augmentations=str(self.cfg.augs.slug),
                        dataset_train_image_size=f"{self.cfg.train.train_image_size}",
                        # Sampling
                        sampler=str(self.cfg.sampler.slug),
                        sampler_num_samples=int(self.cfg.sampler.num_samples),
                        sampler_type=str(self.cfg.sampler.sampler_type),
                        sampler_balance_crowd=bool(self.cfg.sampler.balance_crowd),
                        balance_near_shore=bool(self.cfg.sampler.balance_near_shore),
                        balance_vessel_type=bool(self.cfg.sampler.balance_vessel_type),
                        balance_fishing_type=bool(self.cfg.sampler.balance_fishing_type),
                        balance_location=bool(self.cfg.sampler.balance_location),
                        # Loss
                        loss=str(self.cfg.loss.slug),
                    )
                ),
            )

        return callbacks

    def get_model(self, model_config: DictConfig) -> nn.Module:
        model: nn.Module = get_detection_model(model_config)
        self.box_coder = model.box_coder

        self.master_print("BoxCoder")
        self.master_print(self.box_coder)
        return model


@hydra_dpp_friendly_main(config_path="configs", config_name="multilabel_circle_net")
def main(config: DictConfig) -> None:
    torch.cuda.empty_cache()
    catalyst.utils.set_global_seed(int(config.seed))

    torch.set_anomaly_enabled(config.torch.detect_anomaly)

    torch.backends.cudnn.deterministic = config.torch.deterministic
    torch.backends.cudnn.benchmark = config.torch.benchmark

    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
    torch.backends.cuda.matmul.allow_tf32 = config.torch.cuda_allow_tf32
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = config.torch.cudnn_allow_tf32

    if config.dataset.data_dir is None:
        raise ValueError("--data-dir must be set")

    CenternetPipeline(config).train()


if __name__ == "__main__":
    main()
