import os
from functools import partial
from multiprocessing import Pool
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed
from catalyst.dl import IRunner, Callback, CallbackOrder
from pytorch_toolbelt.datasets import INPUT_IMAGE_KEY
from pytorch_toolbelt.utils import to_numpy
from pytorch_toolbelt.utils.catalyst import get_tensorboard_logger
from pytorch_toolbelt.utils.distributed import (
    reduce_dict_sum,
    is_main_process,
    broadcast_from_master,
    all_gather,
)
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .bboxer import MultilabelCircleNetCoder, MultilabelCircleNetDecodeResult
from .constants import (
    CENTERNET_OUTPUT_OFFSET,
    CENTERNET_OUTPUT_OBJECTNESS_MAP,
    CENTERNET_OUTPUT_DECODED_PREDICTIONS,
    CENTERNET_TARGET_SIZE,
    CENTERNET_TARGET_FISHING_MAP,
    CENTERNET_TARGET_OBJECTNESS_MAP,
    CENTERNET_TARGET_VESSEL_MAP,
    CENTERNET_OUTPUT_SIZE,
    CENTERNET_OUTPUT_FISHING_MAP,
    CENTERNET_OUTPUT_VESSEL_MAP,
)
from ..constants import TARGET_CENTERS_KEY, INPUT_SCENE_ID_KEY, INPUT_SCENE_CROP_KEY
from ..dataset import XView3DataModule
from ..evaluation import apply_thresholds
from ..metric import official_metric_scoring

__all__ = [
    "CircleNetRocAucMetricsCallback",
    "MultilabelCircleNetDetectionMetrics",
    "MultilabelCircleNetDecodeCallback",
]


def _compute_scores_fn(objectness_threshold, predictions, groundtruths, shore_root):
    df = apply_thresholds(predictions, objectness_threshold, None, None)
    summary = official_metric_scoring(df, groundtruths, shore_root)
    summary["objectness_threshold"] = objectness_threshold

    return objectness_threshold, summary


class AbstractObjectDetectionMetricsCallback(Callback):
    groundtruth_df: pd.DataFrame
    predictions: Optional[Dict]
    shore_root: str
    distance_tolerance_meters: float
    shore_tolerance_km: float
    suffix: str
    prefix: str
    all_metrics: bool

    def __init__(
        self,
        groundtruth_df: pd.DataFrame,
        shore_root: str = None,
        prefix="",
        suffix="",
        all_metrics=False,
        objectness_thresholds=(0.45, 0.5, 0.55),
    ):
        super().__init__(CallbackOrder.Metric)
        self.groundtruth_df = groundtruth_df
        self.objectness_thresholds = objectness_thresholds
        self.predictions = None
        self.shore_root = shore_root
        self.suffix = suffix
        self.prefix = prefix
        self.all_metrics = all_metrics

    def on_loader_start(self, state: IRunner):
        self.predictions = {
            "scene_id": [],
            "objectness_p": [],
            "is_vessel_p": [],
            "is_fishing_p": [],
            "vessel_length_m": [],
            "detect_scene_row": [],
            "detect_scene_column": [],
        }

    def decode_objects(self, state):
        raise NotImplementedError

    @torch.no_grad()
    def on_batch_end(self, state: IRunner):
        for centers, objectness, is_vessel, is_fishing, lengths, scene_id in self.decode_objects(state):
            self.predictions["scene_id"].extend([scene_id] * len(centers))
            self.predictions["objectness_p"].extend(objectness)
            self.predictions["is_vessel_p"].extend(is_vessel)
            self.predictions["is_fishing_p"].extend(is_fishing)
            self.predictions["vessel_length_m"].extend(lengths)
            self.predictions["detect_scene_row"].extend(centers[:, 1])
            self.predictions["detect_scene_column"].extend(centers[:, 0])

    @torch.no_grad()
    def on_loader_end(self, state: IRunner):
        predictions = pd.DataFrame.from_dict(reduce_dict_sum(self.predictions))

        thresholds_summary = None
        if is_main_process():

            process_fn = partial(
                _compute_scores_fn,
                predictions=predictions,
                groundtruths=self.groundtruth_df,
                shore_root=self.shore_root,
            )

            if len(self.objectness_thresholds) > 1:
                thresholds_summary = []
                num_workers = 10 if os.name == "nt" else 30
                num_workers = min(len(self.objectness_thresholds), num_workers)

                with Pool(num_workers) as wp:
                    for objectness_threshold, summary_for_threshold in tqdm(
                        wp.imap_unordered(process_fn, self.objectness_thresholds, chunksize=1),
                        total=len(self.objectness_thresholds),
                        desc="Computing scores",
                    ):
                        # Thresholds
                        thresholds_summary.append(summary_for_threshold)

                thresholds_summary = pd.concat(thresholds_summary)
            else:
                objectness_threshold, thresholds_summary = process_fn(self.objectness_thresholds[0])

            thresholds_summary = thresholds_summary.sort_values(by="aggregate", ascending=False).reset_index(drop=True)

            writer: SummaryWriter = get_tensorboard_logger(state)
            writer.add_text("thresholds_summary", text_string=thresholds_summary.to_markdown(), global_step=state.global_epoch)

        thresholds_summary = broadcast_from_master(thresholds_summary)

        for key in {"objectness_threshold", "is_vessel_threshold", "is_fishing_threshold"}:
            state.loader_metrics[f"thresholds/{self.prefix}{key}"] = float(thresholds_summary[key].iloc[0])

        for key in {
            "aggregate",
            "length_acc",
            "loc_fscore",
            "vessel_fscore",
            "fishing_fscore",
            "loc_fscore_shore",
        }:
            state.loader_metrics[f"metrics/{self.prefix}{key}"] = float(thresholds_summary[key].iloc[0])


class MultilabelCircleNetDecodeCallback(Callback):
    box_coder: MultilabelCircleNetCoder

    def __init__(self, box_coder: MultilabelCircleNetCoder):
        super().__init__(CallbackOrder.Metric)
        self.box_coder = box_coder

    @torch.no_grad()
    def on_batch_end(self, runner: "IRunner"):
        box_coder = self.box_coder.box_coder_for_image_size(
            image_size=(
                runner.input[INPUT_IMAGE_KEY].size(2),
                runner.input[INPUT_IMAGE_KEY].size(3),
            )
        )

        # Adaptively compute max objects, limiting it by either 1024 or double the number of boxes in gt
        max_objects = min(2048, max(128, 2 * max([len(labels) for labels in runner.input[TARGET_CENTERS_KEY]])))

        predictions = box_coder.decode(
            objectness_map=runner.output[CENTERNET_OUTPUT_OBJECTNESS_MAP],
            is_vessel_map=runner.output[CENTERNET_OUTPUT_VESSEL_MAP],
            is_fishing_map=runner.output[CENTERNET_OUTPUT_FISHING_MAP],
            length_map=runner.output[CENTERNET_OUTPUT_SIZE],
            offset_map=runner.output.get(CENTERNET_OUTPUT_OFFSET, None),
            max_objects=max_objects,
            apply_activation=True,
        )

        runner.output[CENTERNET_OUTPUT_DECODED_PREDICTIONS] = predictions


class MultilabelCircleNetDetectionMetrics(AbstractObjectDetectionMetricsCallback):
    def __init__(
        self,
        groundtruth_df: pd.DataFrame,
        shore_root=None,
        prefix="",
        suffix="",
        all_metrics=False,
        objectness_thresholds=(0.45, 0.5, 0.55),
    ):
        super().__init__(
            groundtruth_df=groundtruth_df,
            shore_root=shore_root,
            prefix=prefix,
            suffix=suffix,
            all_metrics=all_metrics,
            objectness_thresholds=objectness_thresholds,
        )

    def decode_objects(self, state):
        predictions: MultilabelCircleNetDecodeResult = state.output[CENTERNET_OUTPUT_DECODED_PREDICTIONS]

        batch_size = len(predictions.centers)
        results = []

        for i in range(batch_size):
            (start_row, _), (start_col, _) = state.input[INPUT_SCENE_CROP_KEY][i]
            centers = to_numpy(predictions.centers[i]) + np.array([[start_col, start_row]])
            objectness_p = to_numpy(predictions.scores[i])
            is_vessel_p = to_numpy(predictions.is_vessel[i])
            is_fishing_p = to_numpy(predictions.is_fishing[i])
            lengths = XView3DataModule.decode_lengths(predictions.lengths[i])
            scene_id = state.input[INPUT_SCENE_ID_KEY][i]
            results.append((centers.astype(int), objectness_p, is_vessel_p, is_fishing_p, lengths, scene_id))

        return results


class CircleNetRocAucMetricsCallback(Callback):
    all_objectness_groundtruths = None
    all_objectness_predictions = None

    all_fishing_groundtruths = None
    all_fishing_predictions = None

    all_vessel_groundtruths = None
    all_vessel_predictions = None

    all_length_groundtruths = None
    all_length_predictions = None

    def __init__(self, prefix="metrics", fix_nans: bool = True):
        self.prefix = prefix
        super().__init__(CallbackOrder.Metric)
        self.fix_nans = fix_nans

    def on_loader_start(self, runner: "IRunner"):
        self.all_objectness_groundtruths = []
        self.all_objectness_predictions = []

        self.all_fishing_groundtruths = []
        self.all_fishing_predictions = []

        self.all_vessel_groundtruths = []
        self.all_vessel_predictions = []

        self.all_length_groundtruths = []
        self.all_length_predictions = []

    @torch.no_grad()
    def on_batch_end(self, runner: "IRunner"):
        gt_obj_map = runner.input[CENTERNET_TARGET_OBJECTNESS_MAP].detach().eq(1)
        num_instances = torch.count_nonzero(gt_obj_map)
        if num_instances:
            gt_vessel_map = runner.input[CENTERNET_TARGET_VESSEL_MAP][gt_obj_map]
            gt_fishing_map = runner.input[CENTERNET_TARGET_FISHING_MAP][gt_obj_map]
            gt_sizes = MultilabelCircleNetCoder.log2length(runner.input[CENTERNET_TARGET_SIZE][gt_obj_map])

            # Masks
            gt_vessel_mask = gt_vessel_map.eq(0) | gt_vessel_map.eq(1)
            gt_fishing_mask = gt_fishing_map.eq(0) | gt_fishing_map.eq(1)
            gt_sizes_mask = gt_sizes > 0

            # pred_obj_map = runner.output[CENTERNET_OUTPUT_OBJECTNESS_MAP][gt_obj_map].sigmoid()
            pred_fsh_map = runner.output[CENTERNET_OUTPUT_FISHING_MAP][gt_obj_map].detach().float().sigmoid()
            pred_vsl_map = runner.output[CENTERNET_OUTPUT_VESSEL_MAP][gt_obj_map].detach().float().sigmoid()
            pred_sizes = MultilabelCircleNetCoder.log2length(runner.output[CENTERNET_OUTPUT_SIZE][gt_obj_map].detach())

            self.all_vessel_groundtruths.extend(to_numpy(gt_vessel_map[gt_vessel_mask]).flatten())
            self.all_vessel_predictions.extend(to_numpy(pred_vsl_map[gt_vessel_mask]).flatten())

            self.all_fishing_groundtruths.extend(to_numpy(gt_fishing_map[gt_fishing_mask]).flatten())
            self.all_fishing_predictions.extend(to_numpy(pred_fsh_map[gt_fishing_mask]).flatten())

            self.all_length_groundtruths.extend(to_numpy(gt_sizes[gt_sizes_mask]).flatten())
            self.all_length_predictions.extend(to_numpy(pred_sizes[gt_sizes_mask]).flatten())

    def on_loader_end(self, runner: "IRunner"):
        all_vessel_groundtruths = np.concatenate(all_gather(self.all_vessel_groundtruths))
        all_vessel_predictions = np.concatenate(all_gather(self.all_vessel_predictions))

        all_fishing_groundtruths = np.concatenate(all_gather(self.all_fishing_groundtruths))
        all_fishing_predictions = np.concatenate(all_gather(self.all_fishing_predictions))

        if self.fix_nans:
            all_vessel_predictions[~np.isfinite(all_vessel_predictions)] = 0.5
            all_fishing_predictions[~np.isfinite(all_fishing_predictions)] = 0.5

        vessel_auc = roc_auc_score(y_score=all_vessel_predictions, y_true=all_vessel_groundtruths)
        fishing_auc = roc_auc_score(y_score=all_fishing_predictions, y_true=all_fishing_groundtruths)

        all_length_groundtruths = np.concatenate(all_gather(self.all_length_groundtruths))
        all_length_predictions = np.concatenate(all_gather(self.all_length_predictions))
        length_score = 1.0 - min(1, (np.abs(all_length_predictions - all_length_groundtruths) / all_length_groundtruths).mean())

        runner.loader_metrics[f"{self.prefix}/fishing_auc"] = float(fishing_auc)
        runner.loader_metrics[f"{self.prefix}/vessel_auc"] = float(vessel_auc)
        runner.loader_metrics[f"{self.prefix}/length_score"] = float(length_score)
        runner.loader_metrics[f"{self.prefix}/mean_score"] = (fishing_auc + vessel_auc + length_score) / 3

        if is_main_process():
            logger = get_tensorboard_logger(runner)

            for class_label in range(2):
                p = all_vessel_predictions[all_vessel_groundtruths == class_label]
                if p.any():
                    logger.add_histogram(
                        tag=f"{self.prefix}/is_vessel/{class_label}",
                        values=p,
                        global_step=runner.global_epoch,
                    )

                p = all_fishing_predictions[all_fishing_groundtruths == class_label]
                if p.any():
                    logger.add_histogram(
                        tag=f"{self.prefix}/is_fishing/{class_label}",
                        values=p,
                        global_step=runner.global_epoch,
                    )
