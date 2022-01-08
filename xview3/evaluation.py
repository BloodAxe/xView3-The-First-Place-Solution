import os
from functools import partial
from multiprocessing import Pool
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.distributed import is_main_process
from torch import nn
from tqdm import tqdm

from xview3.centernet.visualization import create_false_color_composite, vis_detections_opencv
from xview3.constants import PIX_TO_M
from xview3.dataset import read_multichannel_image, SigmoidNormalization, XView3DataModule
from xview3.inference import predict_multilabel_scenes
from xview3.metric import official_metric_scoring, official_metric_scoring_per_scene

__all__ = ["apply_thresholds", "compute_optimal_thresholds", "evaluate_on_scenes"]


def apply_thresholds(
    df: pd.DataFrame, objectness_threshold: Optional[float], is_vessel_threshold: Optional[float], is_fishing_threshold: Optional[float]
) -> pd.DataFrame:
    df = df.copy()
    if objectness_threshold is not None:
        df = df[df.objectness_p >= objectness_threshold].copy().reset_index(drop=True)
    if is_vessel_threshold is not None:
        df["is_vessel"] = df["is_vessel_p"] >= is_vessel_threshold
    if is_fishing_threshold is not None:
        df["is_fishing"] = df["is_fishing_p"] >= is_fishing_threshold
    return df


from pytorch_toolbelt.utils import (
    to_numpy,
)


def compute_optimal_thresholds(
    predictions: pd.DataFrame,
    groundtruths: pd.DataFrame,
    shore_root,
    objectness_thresholds=None,
):
    objectness_thresholds = np.linspace(0.4, 0.6, 21) if objectness_thresholds is None else to_numpy(list(objectness_thresholds))
    # objectness_thresholds = np.array([0.4, 0.5, 0.6])
    print("Computing optimal thresholds")
    print("objectness_thresholds", objectness_thresholds)

    compute_scores_fn = partial(
        _compute_scores_fn,
        predictions=predictions,
        groundtruths=groundtruths,
        shore_root=shore_root,
    )

    num_workers = 6 if os.name == "nt" else 30
    num_workers = min(len(objectness_thresholds), num_workers)
    with Pool(num_workers) as wp:
        thresholds_summary = []
        for summary in tqdm(
            wp.imap_unordered(compute_scores_fn, objectness_thresholds, chunksize=1),
            desc="Computing optimal vessel & fishing thresholds",
            total=len(objectness_thresholds),
        ):
            thresholds_summary.append(summary)

    thresholds_summary = pd.concat(thresholds_summary).sort_values(by="aggregate", ascending=False).reset_index(drop=True)
    return thresholds_summary


def _compute_scores_fn(
    payload,
    predictions,
    groundtruths,
    shore_root,
):
    objectness_threshold = payload
    df = apply_thresholds(predictions, objectness_threshold, None, None)
    scores = official_metric_scoring(df, groundtruths, shore_root)
    scores["objectness_threshold"] = objectness_threshold
    return scores


def evaluate_on_scenes(
    model: nn.Module,
    box_coder,
    scenes,
    channels,
    normalization,
    shore_root: str,
    valid_df: pd.DataFrame,
    tile_size: int,
    tile_step: int,
    fp16: bool,
    apply_activation: bool,
    prefix: str,
    suffix: str,
    objectness_thresholds=None,
    output_dir: str = None,
    batch_size: int = 1,
    channels_last: bool = False,
    accumulate_on_gpu: bool = False,
    save_predictions: bool = True,
    max_objects: int = 2048,
    run_evaluation=True,
):
    predictions_dir = output_dir
    os.makedirs(predictions_dir, exist_ok=True)

    multi_score_valid_predictions = predict_multilabel_scenes(
        model=model,
        box_coder=box_coder,
        scenes=scenes,
        channels=channels,
        normalization=normalization,
        objectness_thresholds_lower_bound=0.3,
        output_predictions_dir=predictions_dir,
        accumulate_on_gpu=accumulate_on_gpu,
        fp16=fp16,
        tile_size=tile_size,
        tile_step=tile_step,
        apply_activation=apply_activation,
        batch_size=batch_size,
        max_objects=max_objects,
        save_raw_predictions=save_predictions,
        channels_last=channels_last,
    )

    if run_evaluation and is_main_process():
        multi_score_valid_predictions.to_csv(os.path.join(predictions_dir, f"{prefix}unfiltered_predictions{suffix}.csv"), index=False)

        thresholds_summary = compute_optimal_thresholds(
            predictions=multi_score_valid_predictions,
            groundtruths=valid_df,
            shore_root=shore_root,
            objectness_thresholds=objectness_thresholds,
        )

        print(thresholds_summary.head())
        thresholds_summary.to_csv(os.path.join(predictions_dir, f"{prefix}thresholds_summary{suffix}.csv"), index=False)

        # Visualize predictions using best thresholds
        objectness_threshold = thresholds_summary.loc[thresholds_summary["aggregate"].idxmax(), "objectness_threshold"]
        vessel_threshold = thresholds_summary.loc[thresholds_summary["aggregate"].idxmax(), "is_vessel_threshold"]
        fishing_threshold = thresholds_summary.loc[thresholds_summary["aggregate"].idxmax(), "is_fishing_threshold"]

        scores_per_scene = official_metric_scoring_per_scene(
            multi_score_valid_predictions, valid_df, shore_root, objectness_threshold, vessel_threshold, fishing_threshold
        )
        scores_per_scene.to_csv(os.path.join(predictions_dir, f"{prefix}scores_per_scene{suffix}.csv"), index=False)

        print("Generating visualizations using thresholds", objectness_threshold, vessel_threshold, fishing_threshold)

        predictions = apply_thresholds(multi_score_valid_predictions, objectness_threshold, vessel_threshold, fishing_threshold)

        for scene_path in tqdm(scenes, desc="Making visualizations"):
            scene_id = fs.id_from_fname(scene_path)
            scene_df = predictions[predictions.scene_id == scene_id]

            image = read_multichannel_image(scene_path, ["vv", "vh"])

            normalize = SigmoidNormalization()
            size_down_4 = image["vv"].shape[1] // 4, image["vv"].shape[0] // 4
            image_rgb = create_false_color_composite(
                normalize(image=cv2.resize(image["vv"], dsize=size_down_4, interpolation=cv2.INTER_AREA))["image"],
                normalize(image=cv2.resize(image["vh"], dsize=size_down_4, interpolation=cv2.INTER_AREA))["image"],
            )
            image_rgb[~np.isfinite(image_rgb)] = 0

            scene_predictions = XView3DataModule.get_multilabel_targets_from_df(scene_df)
            centers = (scene_predictions.centers * 0.25).astype(int)

            image_rgb = vis_detections_opencv(
                image_rgb,
                centers=centers,
                lengths=XView3DataModule.decode_lengths(scene_predictions.lengths) / PIX_TO_M,
                is_vessel_vec=scene_predictions.is_vessel,
                is_fishing_vec=scene_predictions.is_fishing,
                scores=np.ones(len(centers)),
                show_title=True,
                alpha=0.1,
            )
            cv2.imwrite(os.path.join(predictions_dir, scene_id + ".jpg"), image_rgb)
