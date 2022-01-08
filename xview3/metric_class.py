import json
import os
from typing import Dict, Union, List

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree, distance_matrix
from sklearn.neighbors import BallTree

from .constants import PIX_TO_M, MAX_OBJECT_LENGTH_M

__all__ = ["Scorer"]

from .metric import (
    calculate_p_r_f,
    compute_vessel_class_performance,
    aggregate_f,
    compute_length_performance,
    compute_fishing_class_performance,
    compute_loc_performance,
)


def get_shoreline_shoreline_contours(shoreline_root, scene_id) -> np.ndarray:
    shoreline_places = [
        f"{shoreline_root}/{scene_id}_shoreline.npy",
        f"{shoreline_root}/train/{scene_id}_shoreline.npy",
        f"{shoreline_root}/validation/{scene_id}_shoreline.npy",
    ]

    shoreline_contours = None
    for shoreline_path in shoreline_places:
        if os.path.isfile(shoreline_path):
            shoreline_contours = np.load(shoreline_path, allow_pickle=True)
            break
    if shoreline_contours is None:
        raise RuntimeError("Could not locate shoreline_contours path")

    contour_points = np.vstack(shoreline_contours)
    return contour_points.reshape((-1, 2))


class Scorer:
    shoreline_trees: Dict[str, KDTree]
    scene_ids: List[str]
    gt_scenes: Dict[str, pd.DataFrame]
    gt_scenes_shore: Dict[str, pd.DataFrame]
    groundtruth_array: Dict[str, np.ndarray]
    gt_shoreline_trees: Dict[str, KDTree]
    leaf_size = 16

    def __init__(
        self,
        groundtruth: pd.DataFrame,
        shoreline_root,
        distance_tolerance_m=200.0,
        shore_tolerance_km=2.0,
    ):
        self.groundtruth = groundtruth
        self.shoreline_root = shoreline_root
        self.shore_tolerance_km = shore_tolerance_km
        self.distance_tolerance_m = distance_tolerance_m

        self.scene_ids = list(groundtruth.scene_id.unique())
        self.shoreline_trees = {}
        self.gt_shoreline_trees = {}
        self.gt_scenes = {}
        self.gt_scenes_shore = {}
        self.groundtruth_array = {}
        self.groundtruth_shore_array = {}

        for scene_id in self.scene_ids:
            # Creating KD trees and computing distance matrix
            contour_points = get_shoreline_shoreline_contours(shoreline_root, scene_id)
            shoreline_kdtree = KDTree(contour_points, leafsize=self.leaf_size) if len(contour_points) else None
            self.shoreline_trees[scene_id] = shoreline_kdtree

            # Create KD tree for objects locations
            scene_df = groundtruth[groundtruth.scene_id == scene_id]
            scene_df_shore = scene_df[scene_df.distance_from_shore_km <= shore_tolerance_km]
            gt_array = np.column_stack((scene_df["detect_scene_row"].values, scene_df["detect_scene_column"].values)).reshape((-1, 2))
            gt_shore_array = np.column_stack(
                (scene_df_shore["detect_scene_row"].values, scene_df_shore["detect_scene_column"].values)
            ).reshape((-1, 2))

            self.gt_scenes[scene_id] = scene_df
            self.gt_scenes_shore[scene_id] = scene_df_shore
            self.groundtruth_array[scene_id] = gt_array
            self.groundtruth_shore_array[scene_id] = gt_shore_array
            # self.gt_shoreline_trees[scene_id] = KDTree(gt_array)

    def compute_loc_performance(self, preds: pd.DataFrame, scene_id):
        pred_array = np.column_stack((preds["detect_scene_row"].values, preds["detect_scene_column"].values)).reshape((-1, 2))
        gt_array = self.groundtruth_array[scene_id]
        gt = self.gt_scenes[scene_id]

        # Building distance matrix using Euclidean distance pixel space
        # multiplied by the UTM resolution (10 m per pixel)
        dist_mat = distance_matrix(pred_array, gt_array, p=2) * PIX_TO_M

        # Using Hungarian matching algorithm to assign lowest-cost gt-pred pairs
        rows, cols = linear_sum_assignment(dist_mat)

        # Recording indices for tp, fp, fn
        tp_inds = [
            {"pred_idx": preds.index[rows[ii]], "gt_idx": gt.index[cols[ii]]}
            for ii in range(len(rows))
            if dist_mat[rows[ii], cols[ii]] < self.distance_tolerance_m
        ]

        tp_inds_wo_low = [
            {"pred_idx": preds.index[rows[ii]], "gt_idx": gt.index[cols[ii]]}
            for ii in range(len(rows))
            if dist_mat[rows[ii], cols[ii]] < self.distance_tolerance_m and gt.confidence.iloc[cols[ii]] != "LOW"
        ]

        tp_pred_inds = [a["pred_idx"] for a in tp_inds]
        tp_gt_inds = [a["gt_idx"] for a in tp_inds]

        fp_inds = [a for a in preds.index if a not in tp_pred_inds]
        fn_inds = [a for a in gt.index if a not in tp_gt_inds and gt.confidence[a] != "LOW"]

        # Making sure each GT is associated with one true positive
        # or is in the false negative bin
        # assert len(gt) == len(fn_inds) + len(tp_inds)

        return tp_inds_wo_low, fp_inds, fn_inds

    def get_shore_preds(self, pred_sc, scene_id, shore_tolerance_km: float):
        if len(pred_sc) == 0:
            return pd.DataFrame()
        predictions = np.array([pred_sc["detect_scene_row"], pred_sc["detect_scene_column"]]).transpose().reshape((-1, 2))

        tree1 = self.shoreline_trees[scene_id]
        tree2 = KDTree(predictions, leafsize=self.leaf_size)
        sdm = tree1.sparse_distance_matrix(tree2, shore_tolerance_km * 1000 / PIX_TO_M, p=2)
        dists = sdm.toarray()

        # Make it so we can use np.min() to find smallest distance b/t each detection and any contour point
        dists[dists == 0] = 9999999
        min_shore_dists = np.min(dists, axis=0)
        close_shore_inds = np.where(min_shore_dists != 9999999)
        df_close = pred_sc.iloc[close_shore_inds]
        return df_close

    def compute_shore_score(self, pred_sc, scene_id):
        # For each scene, compute distances to shore for model predictions, and isolate
        # both predictions and ground truth that are within the appropriate distance
        # from shore.  Note that for the predictions, we include any predictions within
        # shore_tolerance + distance_tolerance/1000.
        gt_sc_shore = self.gt_scenes_shore[scene_id]
        pred_sc_shore = self.get_shore_preds(
            pred_sc,
            scene_id,
            shore_tolerance_km=self.shore_tolerance_km + self.distance_tolerance_m / 1000.0,
        )

        tp_inds_shore, fp_inds_shore, fn_inds_shore = [], [], []
        if (len(gt_sc_shore) > 0) and (len(pred_sc_shore) > 0):
            (
                tp_inds_sc_shore,
                fp_inds_sc_shore,
                fn_inds_sc_shore,
            ) = compute_loc_performance(pred_sc_shore, gt_sc_shore, distance_tolerance=self.distance_tolerance_m)
            tp_inds_shore += tp_inds_sc_shore
            fp_inds_shore += fp_inds_sc_shore
            fn_inds_shore += fn_inds_sc_shore

        return tp_inds_shore, fp_inds_shore, fn_inds_shore

    def compute(self, predictions: pd.DataFrame) -> Dict:
        tp_inds, fp_inds, fn_inds = [], [], []
        tp_inds_shore, fp_inds_shore, fn_inds_shore = [], [], []

        for scene_id in self.scene_ids:
            pred_sc = predictions[predictions["scene_id"] == scene_id]

            tp_inds_sc, fp_inds_sc, fn_inds_sc = self.compute_loc_performance(pred_sc, scene_id)
            tp_inds += tp_inds_sc
            fp_inds += fp_inds_sc
            fn_inds += fn_inds_sc

            # For each scene, compute tp, fp, fn indices by applying the matching algorithm
            # while only considering close-to-shore predictions and detections
            tp_inds_sc_shore, fp_inds_sc_shore, fn_inds_sc_shore = self.compute_shore_score(pred_sc, scene_id)
            tp_inds_shore += tp_inds_sc_shore
            fp_inds_shore += fp_inds_sc_shore
            fn_inds_shore += fn_inds_sc_shore

        # Compute precision, recall, and F1 for maritime object detection
        loc_precision, loc_recall, loc_fscore = calculate_p_r_f(tp_inds, fp_inds, fn_inds)

        if len(
            self.groundtruth[
                (self.groundtruth.scene_id.isin(list(predictions.scene_id.unique())))
                & (self.groundtruth.distance_from_shore_km <= self.shore_tolerance_km)
            ]
        ):
            # Compute precision, recall, F1 for close-to-shore maritime object detection
            loc_precision_shore, loc_recall_shore, loc_fscore_shore = calculate_p_r_f(tp_inds_shore, fp_inds_shore, fn_inds_shore)
        else:
            loc_precision_shore, loc_recall_shore, loc_fscore_shore = 0, 0, 0

        # Getting ground truth vessel indices using is_vessel field in gt
        vessel_inds = self.groundtruth["is_vessel"].isin([True])

        # Getting performance on vessel classification task
        v_tp_inds, v_fp_inds, v_fn_inds, v_tn_inds = compute_vessel_class_performance(
            predictions["is_vessel"].values, self.groundtruth["is_vessel"].values, tp_inds
        )
        vessel_precision, vessel_recall, vessel_fscore = calculate_p_r_f(
            v_tp_inds,
            v_fp_inds,
            v_fn_inds,
        )

        # Getting performance on fishing classification; note that we only consider
        # ground-truth detections that are actually vessels
        f_tp_inds, f_fp_inds, f_fn_inds, f_tn_inds = compute_fishing_class_performance(
            predictions["is_fishing"].values, self.groundtruth["is_fishing"].values, tp_inds, vessel_inds
        )
        fishing_precision, fishing_recall, fishing_fscore = calculate_p_r_f(
            f_tp_inds,
            f_fp_inds,
            f_fn_inds,
        )

        # Computing length estimation performance
        inf_lengths = predictions["vessel_length_m"].values
        gt_lengths = self.groundtruth["vessel_length_m"].values
        length_acc = compute_length_performance(inf_lengths, gt_lengths, tp_inds)

        # Computing normalized aggregate metric
        aggregate = aggregate_f(loc_fscore, length_acc, vessel_fscore, fishing_fscore, loc_fscore_shore)

        # Creating score dictionary
        scores = {
            "loc_fscore": loc_fscore,
            "loc_fscore_shore": loc_fscore_shore,
            "vessel_fscore": vessel_fscore,
            "fishing_fscore": fishing_fscore,
            "length_acc": length_acc,
            "aggregate": aggregate,
        }

        return scores
