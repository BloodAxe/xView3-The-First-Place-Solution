import json
import os
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree, distance_matrix

from .constants import PIX_TO_M, MAX_OBJECT_LENGTH_M

__all__ = ["drop_low_confidence_preds", "official_metric_scoring", "score_thresholded"]


def official_metric_scoring_per_scene(pred, gt, shore_root, objectness_threshold, is_vessel_threshold, is_fishing_threshold):
    scores_per_scene = defaultdict(list)

    for scene_id in gt.scene_id.unique():
        scores = official_metric_scoring(
            pred[pred.scene_id == scene_id].reset_index(drop=True), gt[gt.scene_id == scene_id].reset_index(drop=True), shore_root
        )
        scores["objectness_threshold"] = float(objectness_threshold)

        scores_for_global_thresholds = scores[
            (scores.objectness_threshold == objectness_threshold)
            & (scores.is_vessel_threshold == is_vessel_threshold)
            & (scores.is_fishing_threshold == is_fishing_threshold)
        ]

        local_thresholds_score = scores.loc[scores["aggregate"].idxmax(), "aggregate"]
        local_is_vessel_threshold = scores.loc[scores["aggregate"].idxmax(), "is_vessel_threshold"]
        local_is_fishing_threshold = scores.loc[scores["aggregate"].idxmax(), "is_fishing_threshold"]

        scores_per_scene["scene_id"].append(scene_id)
        scores_per_scene["global_thresholds_score"].append(scores_for_global_thresholds["aggregate"].values[0])
        scores_per_scene["global_is_vessel_threshold"].append(is_vessel_threshold)
        scores_per_scene["global_is_fishing_threshold"].append(is_fishing_threshold)
        scores_per_scene["local_thresholds_score"].append(local_thresholds_score)
        scores_per_scene["local_is_vessel_threshold"].append(local_is_vessel_threshold)
        scores_per_scene["local_is_fishing_threshold"].append(local_is_fishing_threshold)

    return pd.DataFrame.from_dict(scores_per_scene)


def official_metric_scoring(pred, gt, shore_root) -> pd.DataFrame:
    inference = drop_low_confidence_preds(pred, gt, distance_tolerance=200, costly_dist=True)
    ground_truth = gt[gt["confidence"].isin(["HIGH", "MEDIUM"])].reset_index(drop=True)

    scores = score_multithreshold(inference, ground_truth, shore_root, distance_tolerance=200, shore_tolerance=2, costly_dist=True)
    return scores


def drop_low_confidence_preds(pred, gt, distance_tolerance=200, costly_dist=False):
    """
    Matches detections in a predictions dataframe to a ground truth data frame and isolate the low confidence matches

    Args:
        preds (pd.DataFrame): contains inference results for a
            single scene
        gt (pd.DataFrame): contains ground truth labels for a single
            scene
        distance_tolerance (int, optional): Maximum distance
            for valid detection. Defaults to 200.
        costly_dist (bool): whether to assign 9999999 to entries in the
            distance metrics greater than distance_tolerance; defaults to False

    Returns:
        df_out (pd.DataFrame): preds dataframe without the low confidence matches
    """

    low_inds = []

    # For each scene, obtain the tp, fp, and fn indices for maritime
    # object detection in the *global* pred and gt dataframes
    for scene_id in gt["scene_id"].unique():
        pred_sc = pred[pred["scene_id"] == scene_id]
        gt_sc = gt[gt["scene_id"] == scene_id]
        low_inds_scene = match_low_confidence_preds(pred_sc, gt_sc, distance_tolerance=distance_tolerance, costly_dist=costly_dist)

        low_inds += low_inds_scene

    # Check matched pairs came from "LOW" labels
    for pair in low_inds:
        assert gt.iloc[pair["gt_idx"]]["confidence"] == "LOW", f"Index {pair['gt_idx']} is {gt.iloc[pair['gt_idx']]['confidence']}"

    low_pred_inds = [a["pred_idx"] for a in low_inds]

    df_out = pred.drop(index=low_pred_inds)
    df_out = df_out.reset_index()
    return df_out


def match_low_confidence_preds(preds, gt, distance_tolerance=200, costly_dist=False):
    """
    Matches detections in a predictions dataframe to a ground truth data frame and isolate the low confidence matches

    Args:
        preds (pd.DataFrame): contains inference results for a
            single scene
        gt (pd.DataFrame): contains ground truth labels for a single
            scene
        distance_tolerance (int, optional): Maximum distance
            for valid detection. Defaults to 200.
        costly_dist (bool): whether to assign 9999999 to entries in the
            distance metrics greater than distance_tolerance; defaults to False

    Returns:
        low_inds (list, int): list of indices for the preds dataframe that are
            associated as (1) correct detection in the *global* preds dataframe; (2) low confidence in the corresponding gt dataframe
    """

    # Getting pixel-level predicted and ground-truth detections
    pred_array = np.array(list(zip(preds["detect_scene_row"], preds["detect_scene_column"]))).reshape((-1, 2))
    gt_array = np.array(list(zip(gt["detect_scene_row"], gt["detect_scene_column"]))).reshape((-1, 2))

    # Getting a list of index with LOW in the ground truth dataframe
    low_gt_inds = list(gt[gt["confidence"] == "LOW"].index)

    # Building distance matrix using Euclidean distance pixel space
    # multiplied by the UTM resolution (10 m per pixel)
    dist_mat = distance_matrix(pred_array, gt_array, p=2) * PIX_TO_M
    if costly_dist:
        dist_mat[dist_mat > distance_tolerance] = 9999999 * PIX_TO_M

    # Using Hungarian matching algorithm to assign lowest-cost gt-pred pairs
    rows, cols = linear_sum_assignment(dist_mat)

    low_inds = [
        {"pred_idx": preds.index[rows[ii]], "gt_idx": gt.index[cols[ii]]}
        for ii in range(len(rows))
        if (dist_mat[rows[ii], cols[ii]] < distance_tolerance) and (gt.index[cols[ii]] in low_gt_inds)
    ]

    return low_inds


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

    if len(shoreline_contours):
        contour_points = np.vstack(shoreline_contours)
        return contour_points.reshape((-1, 2))
    else:
        return np.array([]).reshape((-1, 2))


def get_shore_preds(df, shoreline_root, scene_id, shore_tolerance_km):
    """
    Getting detections that are close to the shoreline

    Args:
        df (pd.DataFrame): dataframe containing detections
        shoreline_root (str): path to shoreline contour files
        scene_id (str): scene_id
        shore_tolerance_km (float): "close to shore" tolerance in km

    Returns:
        df_close (pd.DataFrame): subset of df containing only detections close to shore
    """
    # Loading shoreline contours for distance-to-shore calculation
    shoreline_contours = get_shoreline_shoreline_contours(shoreline_root, scene_id)

    # If there are no shorelines in the scene
    if len(shoreline_contours) == 0 or len(df) == 0:
        return pd.DataFrame()

    # Creating KD trees and computing distance matrix
    tree1 = KDTree(np.array(shoreline_contours))
    tree2 = KDTree(np.array([df["detect_scene_row"], df["detect_scene_column"]]).transpose())
    sdm = tree1.sparse_distance_matrix(tree2, shore_tolerance_km * 1000 / PIX_TO_M, p=2)
    dists = sdm.toarray()

    # Make it so we can use np.min() to find smallest distance b/t each detection and any contour point
    dists[dists == 0] = 9999999
    min_shore_dists = np.min(dists, axis=0)
    close_shore_inds = np.where(min_shore_dists != 9999999)
    df_close = df.iloc[close_shore_inds]
    return df_close


def compute_loc_performance(preds, gt, distance_tolerance=200, costly_dist=False):
    """
    Computes maritime object detection performance from a prediction
    dataframe and a ground truth datafr

    Args:
        preds (pd.DataFrame): contains inference results for a
            single scene
        gt (pd.DataFrame): contains ground truth labels for a single
            scene
        distance_tolerance (int, optional): Maximum distance
            for valid detection. Defaults to 200.
        costly_dist (bool): whether to assign 9999999 to entries in the distance metrics greater than distance_tolerance; defaults to False

    Returns:
        tp_ind (list, dict): list of dicts with keys 'pred_idx', 'gt_idx';
            values for each are the indices preds and gt that are
            associated as correct detection in the *global* preds and
            gt dataframes
        fp_ind (list): list of indices in the *global* preds dataframe
            that are not assigned to a gt detection by the matching
        fn_ind (list): list of indices in the *global* gt dataframe
            that do not match any detection in pred within dist_tol
    """

    # Getting pixel-level predicted and ground-truth detections
    pred_array = np.array(list(zip(preds["detect_scene_row"], preds["detect_scene_column"]))).reshape((-1, 2))
    gt_array = np.array(list(zip(gt["detect_scene_row"], gt["detect_scene_column"]))).reshape((-1, 2))

    # Building distance matrix using Euclidean distance pixel space
    # multiplied by the UTM resolution (10 m per pixel)
    dist_mat = distance_matrix(pred_array, gt_array, p=2) * PIX_TO_M
    if costly_dist:
        dist_mat[dist_mat > distance_tolerance] = 9999999 * PIX_TO_M

    # Using Hungarian matching algorithm to assign lowest-cost gt-pred pairs
    rows, cols = linear_sum_assignment(dist_mat)

    # Recording indices for tp, fp, fn
    tp_inds = [
        {"pred_idx": preds.index[rows[ii]], "gt_idx": gt.index[cols[ii]]}
        for ii in range(len(rows))
        if dist_mat[rows[ii], cols[ii]] < distance_tolerance
    ]
    tp_pred_inds = [a["pred_idx"] for a in tp_inds]
    tp_gt_inds = [a["gt_idx"] for a in tp_inds]

    fp_inds = [a for a in preds.index if a not in tp_pred_inds]
    fn_inds = [a for a in gt.index if a not in tp_gt_inds]

    # Making sure each GT is associated with one true positive
    # or is in the false negative bin
    assert len(gt) == len(fn_inds) + len(tp_inds)

    return tp_inds, fp_inds, fn_inds


def compute_vessel_class_performance(preds, gt, tp_inds):
    """
    Identify tp, tn, fp, and fn indices for vessel classification task

    Args:
        preds ([pd.Dataframe)): contains inference results for all scenes
        gt (pd.Dataframe): contains ground truth labels for all scenes
        tp_inds (list, dict): List of dicts output from compute_loc_performance
            containing indices of true positive detection pairs in the global preds
            and gt dataframes

    Returns:
        c_tp_inds: list of dicts from tp_inds where ground truth and
            model output both have a True 'is_vessel' label
        c_tn_inds: list of dicts from tp_inds where ground truth and
            model output both have a False 'is_vessel' label
        c_fp_inds: list of dicts from tp_inds where ground truth
            'is_vessel' label is False but model output is True
        c_fn_inds: list of dicts from tp_inds where ground truth
            'is_vessel' output is True but model output is False
    """
    c_tp_inds = []
    c_fp_inds = []
    c_fn_inds = []
    c_tn_inds = []
    # For every box where you have a matching detection, do the labels match?
    for pair in tp_inds:
        # Making sure we only use valid GTs
        if isinstance(gt[pair["gt_idx"]], float):
            if np.isnan(gt[pair["gt_idx"]]):
                continue
        if preds[pair["pred_idx"]] == gt[pair["gt_idx"]]:
            if gt[pair["gt_idx"]]:
                c_tp_inds.append(pair)
            else:
                c_tn_inds.append(pair)
        else:
            if gt[pair["gt_idx"]]:
                c_fn_inds.append(pair)
            elif gt[pair["gt_idx"]] == False:
                c_fp_inds.append(pair)

    return c_tp_inds, c_fp_inds, c_fn_inds, c_tn_inds


def compute_fishing_class_performance(preds, gt, tp_inds, vessel_inds):
    """
    Identify tp, tn, fp, and fn indices for fishing classification task

    Args:
        preds ([pd.Dataframe)): contains inference results for all scenes
        gt (pd.Dataframe): contains ground truth labels for all scenes
        tp_inds (list, dict): List of dicts output from compute_loc_performance
            containing indices of true positive detection pairs in the global preds
            and gt dataframes

    Returns:
        c_tp_inds: list of dicts from tp_inds where ground truth and
            model output both have a True 'is_fishing' label
        c_tn_inds: list of dicts from tp_inds where ground truth and
            model output both have a False 'is_fishing' label
        c_fp_inds: list of dicts from tp_inds where ground truth
            'is_fishing' label is False but model output is True
        c_fn_inds: list of dicts from tp_inds where ground truth
            'is_fishing' output is True but model output is False
    """
    c_tp_inds = []
    c_fp_inds = []
    c_fn_inds = []
    c_tn_inds = []
    # For every box where you have a matching detection, do the labels match?
    for pair in tp_inds:
        if vessel_inds is not None:
            if not vessel_inds[pair["gt_idx"]]:
                # print('Skipping non-vessels')
                continue
        # Making sure we only use valid GTs
        if isinstance(gt[pair["gt_idx"]], float):
            if np.isnan(gt[pair["gt_idx"]]):
                continue
        if preds[pair["pred_idx"]] == gt[pair["gt_idx"]]:
            if gt[pair["gt_idx"]]:
                c_tp_inds.append(pair)
            else:
                c_tn_inds.append(pair)
        else:
            if gt[pair["gt_idx"]]:
                c_fn_inds.append(pair)
            elif gt[pair["gt_idx"]] == False:
                c_fp_inds.append(pair)

    return c_tp_inds, c_fp_inds, c_fn_inds, c_tn_inds


def compute_length_performance(preds, gt, tp_inds):
    """
    Compute aggregate percent error for vessel size estimation

    Args:
        preds ([pd.Dataframe)): contains inference results for all scenes
        gt (pd.Dataframe): contains ground truth labels for all scenes
        tp_inds (list, dict): List of dicts output from compute_loc_performance
            containing indices of true positive detection pairs in the global preds
            and gt dataframes

    Returns:
        length_performance (float): aggregate percent error for vessel length estimation
    """
    pct_error = 0.0
    num_valid_gt = 0.0

    for pair in tp_inds:
        if isinstance(gt[pair["gt_idx"]], float):
            if np.isnan(gt[pair["gt_idx"]]):
                continue
        # The longest vessel in the world was 458m, and fixed-infrastructure is smaller.
        # To bound the maximum possible error, we cap preds and ground truth
        # at MAX_OBJECT_LENGTH_M, which by default is 500m
        gt_pred = min(gt[pair["gt_idx"]], MAX_OBJECT_LENGTH_M)
        inf_pred = min(preds[pair["pred_idx"]], MAX_OBJECT_LENGTH_M)
        pct_error += np.abs(inf_pred - gt_pred) / gt_pred
        num_valid_gt += 1

    if num_valid_gt == 0:
        length_performance = 0
    else:
        length_performance = 1.0 - min((pct_error / num_valid_gt), 1.0)

    return length_performance


def calculate_p_r_f(tp_inds, fp_inds, fn_inds):
    """
    Compute precision, recall, and F1 score

    Args:
        tp_inds (list, dict): list of dicts with keys 'pred_idx', 'gt_idx';
            values for each are the indices preds and gt that are
            associated as correct detection in the *global* preds and
            gt dataframes
        fp_inds (list): list of indices in the *global* preds dataframe
            that are not assigned to a gt detection by the matching
        fn_inds (list): list of indices in the *global* gt dataframe
            that do not match any detection in pred within dist_tol

    Returns:
        precision (float): precision score
        recall (float): recall score
        fscore (float): f1 score
    """
    try:
        precision = len(tp_inds) / (len(tp_inds) + len(fp_inds))
    except ZeroDivisionError:
        precision = 0
    try:
        recall = len(tp_inds) / (len(tp_inds) + len(fn_inds))
    except ZeroDivisionError:
        recall = 0
    try:
        fscore = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        fscore = 0

    if precision == np.nan or recall == np.nan or fscore == np.nan:
        return 0, 0, 0
    else:
        return precision, recall, fscore


def aggregate_f(loc_fscore, length_acc, vessel_fscore, fishing_fscore, loc_fscore_shore):
    """
    Compute aggregate metric for xView3 scoring


    Args:
        loc_fscore (float): F1 score for overall maritime object detection
        length_acc (float): Aggregate percent error for vessel length estimation
        vessel_fscore (float): F1 score for vessel vs. non-vessel task
        fishing_fscore (float): F1 score for fishing vessel vs. non-fishing vessel task
        loc_fscore_shore (float): F1 score for close-to-shore maritime object detection

    Returns:
        aggregate (float): aggregate metric for xView3 scoring
    """

    # Note: should be between zero and one, and score should be heavily weighted on
    # overall maritime object detection!
    aggregate = loc_fscore * (1 + length_acc + vessel_fscore + fishing_fscore + loc_fscore_shore) / 5

    return aggregate


def score_multithreshold(pred, gt, shore_root, distance_tolerance=200, shore_tolerance=2, costly_dist=False) -> pd.DataFrame:
    """Compute xView3 aggregate score from

    Args:
        pred ([pd.Dataframe)): contains inference results for all scenes
        gt (pd.Dataframe): contains ground truth labels for all scenes]
        shoreline_root (str): path to shoreline contour files
        distance_tolerance (float): Maximum distance
            for valid detection. Defaults to 200.
        shore_tolerance (float): "close to shore" tolerance in km; defaults to 2
        costly_dist (bool): whether to assign 9999999 to entries in the distance metrics greater than distance_tolerance; defaults to False

    Returns:
        scores (dict): dictionary containing aggregate xView score and
            all constituent scores
    """

    tp_inds, fp_inds, fn_inds = [], [], []

    # For each scene, obtain the tp, fp, and fn indices for maritime
    # object detection in the *global* pred and gt dataframes
    for scene_id in gt["scene_id"].unique():
        pred_sc = pred[pred["scene_id"] == scene_id]
        gt_sc = gt[gt["scene_id"] == scene_id]
        (
            tp_inds_sc,
            fp_inds_sc,
            fn_inds_sc,
        ) = compute_loc_performance(pred_sc, gt_sc, distance_tolerance=distance_tolerance, costly_dist=costly_dist)

        tp_inds += tp_inds_sc
        fp_inds += fp_inds_sc
        fn_inds += fn_inds_sc

    # Compute precision, recall, and F1 for maritime object detection
    loc_precision, loc_recall, loc_fscore = calculate_p_r_f(tp_inds, fp_inds, fn_inds)

    # Allowing code to be run without shore data -- will output 0 for these scores
    if shore_tolerance and shore_root:
        # For each scene, compute distances to shore for model predictions, and isolate
        # both predictions and ground truth that are within the appropriate distance
        # from shore.  Note that for the predictions, we include any predictions within
        # shore_tolerance + distance_tolerance/1000.
        tp_inds_shore, fp_inds_shore, fn_inds_shore = [], [], []
        for scene_id in gt["scene_id"].unique():
            pred_sc = pred[pred["scene_id"] == scene_id]
            gt_sc_shore = gt[(gt["scene_id"] == scene_id) & (gt["distance_from_shore_km"] <= shore_tolerance)]
            pred_sc_shore = get_shore_preds(
                pred_sc,
                shore_root,
                scene_id,
                shore_tolerance + distance_tolerance / 1000,
            )
            # print(
            #    f"{len(gt_sc_shore)} ground truth, {len(pred_sc_shore)} predictions close to shore"
            # )
            # For each scene, compute tp, fp, fn indices by applying the matching algorithm
            # while only considering close-to-shore predictions and detections
            if (len(gt_sc_shore) > 0) and (len(pred_sc_shore) > 0):
                (
                    tp_inds_sc_shore,
                    fp_inds_sc_shore,
                    fn_inds_sc_shore,
                ) = compute_loc_performance(pred_sc_shore, gt_sc_shore, distance_tolerance=distance_tolerance, costly_dist=costly_dist)
                tp_inds_shore += tp_inds_sc_shore
                fp_inds_shore += fp_inds_sc_shore
                fn_inds_shore += fn_inds_sc_shore

        if len(gt[(gt["scene_id"].isin(list(pred["scene_id"].unique()))) & (gt["distance_from_shore_km"] <= shore_tolerance)]) > 0:
            # Compute precision, recall, F1 for close-to-shore maritime object detection
            loc_precision_shore, loc_recall_shore, loc_fscore_shore = calculate_p_r_f(tp_inds_shore, fp_inds_shore, fn_inds_shore)
        else:
            loc_precision_shore, loc_recall_shore, loc_fscore_shore = 0, 0, 0
    else:
        loc_precision_shore, loc_recall_shore, loc_fscore_shore = 0, 0, 0

    # Getting ground truth vessel indices using is_vessel field in gt
    vessel_inds = gt["is_vessel"].isin([True])

    thresholds = np.linspace(0.0, 1.0, num=50)

    # Getting performance on vessel classification task
    vessel_fscores = []
    for threshold in thresholds:
        v_tp_inds, v_fp_inds, v_fn_inds, v_tn_inds = compute_vessel_class_performance(
            pred["is_vessel_p"].values >= threshold, gt["is_vessel"].values, tp_inds
        )
        vessel_precision, vessel_recall, vessel_fscore = calculate_p_r_f(
            v_tp_inds,
            v_fp_inds,
            v_fn_inds,
        )
        vessel_fscores.append(vessel_fscore)

    fishing_fscores = []
    # Getting performance on fishing classification; note that we only consider
    # ground-truth detections that are actually vessels
    for threshold in thresholds:
        f_tp_inds, f_fp_inds, f_fn_inds, f_tn_inds = compute_fishing_class_performance(
            pred["is_fishing_p"].values >= threshold, gt["is_fishing"].values, tp_inds, vessel_inds
        )
        fishing_precision, fishing_recall, fishing_fscore = calculate_p_r_f(
            f_tp_inds,
            f_fp_inds,
            f_fn_inds,
        )
        fishing_fscores.append(fishing_fscore)

    # Computing length estimation performance
    inf_lengths = pred["vessel_length_m"].tolist()
    gt_lengths = gt["vessel_length_m"].tolist()
    length_acc = compute_length_performance(inf_lengths, gt_lengths, tp_inds)

    scores = defaultdict(list)

    for vessel_threshold, vessel_fscore in zip(thresholds, vessel_fscores):
        for fishing_threshold, fishing_fscore in zip(thresholds, fishing_fscores):
            # Computing normalized aggregate metric
            aggregate = aggregate_f(loc_fscore, length_acc, vessel_fscore, fishing_fscore, loc_fscore_shore)

            # Creating score dictionary
            scores["loc_fscore"].append(loc_fscore)
            scores["loc_fscore_shore"].append(loc_fscore_shore)
            scores["vessel_fscore"].append(vessel_fscore)
            scores["fishing_fscore"].append(fishing_fscore)
            scores["length_acc"].append(length_acc)
            scores["aggregate"].append(aggregate)
            scores["is_vessel_threshold"].append(vessel_threshold)
            scores["is_fishing_threshold"].append(fishing_threshold)

    return pd.DataFrame.from_dict(scores)


def score_thresholded(pred, gt, shore_root, distance_tolerance=200, shore_tolerance=2, costly_dist=False) -> Dict[str, float]:
    """Compute xView3 aggregate score from

    Args:
        pred ([pd.Dataframe)): contains inference results for all scenes
        gt (pd.Dataframe): contains ground truth labels for all scenes]
        shoreline_root (str): path to shoreline contour files
        distance_tolerance (float): Maximum distance
            for valid detection. Defaults to 200.
        shore_tolerance (float): "close to shore" tolerance in km; defaults to 2
        costly_dist (bool): whether to assign 9999999 to entries in the distance metrics greater than distance_tolerance; defaults to False

    Returns:
        scores (dict): dictionary containing aggregate xView score and
            all constituent scores
    """

    tp_inds, fp_inds, fn_inds = [], [], []

    # For each scene, obtain the tp, fp, and fn indices for maritime
    # object detection in the *global* pred and gt dataframes
    for scene_id in gt["scene_id"].unique():
        pred_sc = pred[pred["scene_id"] == scene_id]
        gt_sc = gt[gt["scene_id"] == scene_id]
        (
            tp_inds_sc,
            fp_inds_sc,
            fn_inds_sc,
        ) = compute_loc_performance(pred_sc, gt_sc, distance_tolerance=distance_tolerance, costly_dist=costly_dist)

        tp_inds += tp_inds_sc
        fp_inds += fp_inds_sc
        fn_inds += fn_inds_sc

    # Compute precision, recall, and F1 for maritime object detection
    loc_precision, loc_recall, loc_fscore = calculate_p_r_f(tp_inds, fp_inds, fn_inds)

    # Allowing code to be run without shore data -- will output 0 for these scores
    if shore_tolerance and shore_root:
        # For each scene, compute distances to shore for model predictions, and isolate
        # both predictions and ground truth that are within the appropriate distance
        # from shore.  Note that for the predictions, we include any predictions within
        # shore_tolerance + distance_tolerance/1000.
        tp_inds_shore, fp_inds_shore, fn_inds_shore = [], [], []
        for scene_id in gt["scene_id"].unique():
            pred_sc = pred[pred["scene_id"] == scene_id]
            gt_sc_shore = gt[(gt["scene_id"] == scene_id) & (gt["distance_from_shore_km"] <= shore_tolerance)]
            pred_sc_shore = get_shore_preds(
                pred_sc,
                shore_root,
                scene_id,
                shore_tolerance + distance_tolerance / 1000,
            )
            # print(
            #    f"{len(gt_sc_shore)} ground truth, {len(pred_sc_shore)} predictions close to shore"
            # )
            # For each scene, compute tp, fp, fn indices by applying the matching algorithm
            # while only considering close-to-shore predictions and detections
            if (len(gt_sc_shore) > 0) and (len(pred_sc_shore) > 0):
                (
                    tp_inds_sc_shore,
                    fp_inds_sc_shore,
                    fn_inds_sc_shore,
                ) = compute_loc_performance(pred_sc_shore, gt_sc_shore, distance_tolerance=distance_tolerance, costly_dist=costly_dist)
                tp_inds_shore += tp_inds_sc_shore
                fp_inds_shore += fp_inds_sc_shore
                fn_inds_shore += fn_inds_sc_shore

        if len(gt[(gt["scene_id"].isin(list(pred["scene_id"].unique()))) & (gt["distance_from_shore_km"] <= shore_tolerance)]) > 0:
            # Compute precision, recall, F1 for close-to-shore maritime object detection
            loc_precision_shore, loc_recall_shore, loc_fscore_shore = calculate_p_r_f(tp_inds_shore, fp_inds_shore, fn_inds_shore)
        else:
            loc_precision_shore, loc_recall_shore, loc_fscore_shore = 0, 0, 0
    else:
        loc_precision_shore, loc_recall_shore, loc_fscore_shore = 0, 0, 0

    # Getting ground truth vessel indices using is_vessel field in gt
    vessel_inds = gt["is_vessel"].isin([True])

    # Getting performance on vessel classification task
    v_tp_inds, v_fp_inds, v_fn_inds, v_tn_inds = compute_vessel_class_performance(
        pred["is_vessel"].values, gt["is_vessel"].values, tp_inds
    )
    vessel_precision, vessel_recall, vessel_fscore = calculate_p_r_f(
        v_tp_inds,
        v_fp_inds,
        v_fn_inds,
    )

    # Getting performance on fishing classification; note that we only consider
    # ground-truth detections that are actually vessels
    f_tp_inds, f_fp_inds, f_fn_inds, f_tn_inds = compute_fishing_class_performance(
        pred["is_fishing"].values, gt["is_fishing"].values, tp_inds, vessel_inds
    )
    fishing_precision, fishing_recall, fishing_fscore = calculate_p_r_f(
        f_tp_inds,
        f_fp_inds,
        f_fn_inds,
    )

    # Computing length estimation performance
    inf_lengths = pred["vessel_length_m"].tolist()
    gt_lengths = gt["vessel_length_m"].tolist()
    length_acc = compute_length_performance(inf_lengths, gt_lengths, tp_inds)

    # Computing normalized aggregate metric
    aggregate = aggregate_f(loc_fscore, length_acc, vessel_fscore, fishing_fscore, loc_fscore_shore)

    # Creating score dictionary
    scores = {}
    scores["loc_fscore"] = loc_fscore
    scores["loc_fscore_shore"] = loc_fscore_shore
    scores["vessel_fscore"] = vessel_fscore
    scores["fishing_fscore"] = fishing_fscore
    scores["length_acc"] = length_acc
    scores["aggregate"] = aggregate

    return scores


def main(args):
    # Print flags for log
    print(f"--score_all: {args.score_all}")
    print(f"--costly_dist: {args.costly_dist}")
    print(f"--drop_low_detect: {args.drop_low_detect}")

    # Read in inference and ground truth detection files
    inference = pd.read_csv(args.inference_file, index_col=False)
    ground_truth = pd.read_csv(args.label_file, index_col=False)

    # If a scene_id list is provided, run only for that scene; otherwise,
    # use all scenes in ground truth
    if args.scene_id is not None:
        inference = inference[inference["scene_id"] == args.scene_id].reset_index()
        ground_truth = ground_truth[ground_truth["scene_id"] == args.scene_id].reset_index()
    else:
        ground_truth = ground_truth[ground_truth["scene_id"].isin(inference["scene_id"].unique())].reset_index()

    # By default we only score on high and medium confidence labels
    if not args.score_all:
        if args.drop_low_detect:
            inference = drop_low_confidence_preds(
                inference, ground_truth, distance_tolerance=args.distance_tolerance, costly_dist=args.costly_dist
            )
        ground_truth = ground_truth[ground_truth["confidence"].isin(["HIGH", "MEDIUM"])].reset_index()

    out = score_multithreshold(
        inference,
        ground_truth,
        args.shore_root,
        args.distance_tolerance,
        args.shore_tolerance,
        args.costly_dist,
    )
    print(out)
    with open(args.output, "w") as fl:
        json.dump(out, fl)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scoring xView3 model.")
    parser.add_argument("--scene_id", help="Scene ID to run evaluations for", default=None)
    parser.add_argument("--inference_file", help="Path to the predictions CSV")
    parser.add_argument("--label_file", help="Path to the xView3 label CSV")
    parser.add_argument("--output", help="Path to output file -- should be .json")
    parser.add_argument("--distance_tolerance", help="Distance tolerance for detection in m", type=int)
    parser.add_argument(
        "--shore_tolerance",
        default=0,
        help="Distance from shore tolerance in km",
        type=int,
    )
    parser.add_argument(
        "--shore_root",
        type=str,
        default=None,
        help="Directory with .npy files containing shore arrays",
    )
    parser.add_argument(
        "--score_all",
        action=argparse.BooleanOptionalAction,
        help="Whether or not to score against all ground truth labels (inclusive of low confidence labels).",
    )
    parser.add_argument(
        "--drop_low_detect",
        action=argparse.BooleanOptionalAction,
        help="Whether or not to drop predictions that are matched to low confidence labels.",
    )
    parser.add_argument(
        "--costly_dist",
        action=argparse.BooleanOptionalAction,
        help="Whether or not to assign a large number (9999999) to distances greater than the distance tolerance threshold.",
    )

    args = parser.parse_args()

    main(args)
