from functools import partial
from multiprocessing import Pool
from typing import Dict

from xview3.metric import (
    aggregate_f,
    calculate_p_r_f,
    get_shore_preds,
    compute_vessel_class_performance,
    compute_length_performance,
    compute_fishing_class_performance,
    compute_loc_performance,
)

__all__ = ["score_mp"]


def _compute_loc_performance_mp(args, distance_tolerance):
    preds, gt = args
    return compute_loc_performance(preds, gt, distance_tolerance)


def _get_shore_preds_and_compute_loc_performance(args, shore_root, shore_tolerance, distance_tolerance):
    pred_sc, gt_sc_shore, scene_id = args
    pred_sc_shore = get_shore_preds(
        pred_sc,
        shore_root,
        scene_id,
        shore_tolerance + distance_tolerance / 1000,
    )

    # print(f"{len(gt_sc_shore)} ground truth, {len(pred_sc_shore)} predictions close to shore")

    # For each scene, compute tp, fp, fn indices by applying the matching algorithm
    # while only considering close-to-shore predictions and detections
    if (len(gt_sc_shore) > 0) and (len(pred_sc_shore) > 0):
        return compute_loc_performance(pred_sc_shore, gt_sc_shore, distance_tolerance_m=distance_tolerance)

    return [], [], []


def score_mp(pred, gt, shore_root, distance_tolerance=200, shore_tolerance=2, workers: int = 8) -> Dict[str, float]:
    """Compute xView3 aggregate score from

    Args:
        pred ([pd.Dataframe)): contains inference results for all scenes
        gt (pd.Dataframe): contains ground truth labels for all scenes]
        shoreline_root (str): path to shoreline contour files
        distance_tolerance (float): Maximum distance
            for valid detection. Defaults to 200.
        shore_tolerance (float): "close to shore" tolerance in km; defaults to 2

    Returns:
        scores (dict): dictionary containing aggregate xView score and
            all constituent scores
    """
    tp_inds, fp_inds, fn_inds = [], [], []

    compute_loc_performance_mp = partial(_compute_loc_performance_mp, distance_tolerance=distance_tolerance)

    get_shore_preds_and_compute_loc_performance = partial(
        _get_shore_preds_and_compute_loc_performance,
        shore_root=shore_root,
        shore_tolerance=shore_tolerance,
        distance_tolerance=distance_tolerance,
    )

    unique_scenes = gt["scene_id"].unique()
    wp = Pool(min(len(unique_scenes), workers))

    # For each scene, obtain the tp, fp, and fn indices for maritime
    # object detection in the *global* pred and gt dataframes
    payload = []
    for scene_id in unique_scenes:
        pred_sc = pred[pred["scene_id"] == scene_id]
        gt_sc = gt[gt["scene_id"] == scene_id]
        payload.append((pred_sc, gt_sc))

    for tp_inds_sc, fp_inds_sc, fn_inds_sc in wp.imap_unordered(compute_loc_performance_mp, payload, chunksize=1):
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

        payload = []

        for scene_id in unique_scenes:
            pred_sc = pred[pred["scene_id"] == scene_id]
            gt_sc_shore = gt[(gt["scene_id"] == scene_id) & (gt["distance_from_shore_km"] <= shore_tolerance)]
            payload.append((pred_sc, gt_sc_shore, scene_id))

        for tp_inds_sc_shore, fp_inds_sc_shore, fn_inds_sc_shore in wp.imap_unordered(
            get_shore_preds_and_compute_loc_performance, payload, chunksize=1
        ):
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
    scores = {
        "loc_fscore": loc_fscore,
        "loc_fscore_shore": loc_fscore_shore,
        "vessel_fscore": vessel_fscore,
        "fishing_fscore": fishing_fscore,
        "length_acc": length_acc,
        "aggregate": aggregate,
        # Precision & Recall scores
        "loc_precision": loc_precision,
        "loc_recall": loc_recall,
        "vessel_precision": vessel_precision,
        "vessel_recall": vessel_recall,
        "fishing_precision": fishing_precision,
        "fishing_recall": fishing_recall,
    }

    wp.close()
    return scores
