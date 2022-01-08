import argparse
import os
from typing import Optional, List, Union

import numpy as np
import pandas as pd
import torch
from pytorch_toolbelt.utils.distributed import all_gather

from xview3 import *
from xview3.centernet.models.inference import get_box_coder_from_model
from xview3.inference import (
    model_from_checkpoint,
    wrap_multilabel_model_with_tta,
    predict_multilabel_scenes,
)

TTA_BATCH_SIZE_DIVISOR = {
    "d4": 8,
    "d2": 4,
    "flips": 3,
}


@torch.no_grad()
def run_multilabel_predict(
    checkpoint_fname: str,
    output_dir: str,
    data_dir: str,
    tta_mode: Optional[str],
    objectness_threshold: Union[float, List[float]],
    vessel_threshold: Union[float, List[float]],
    fishing_threshold: Union[float, List[float]],
    save_predictions=True,
    batch_size=2,
):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = torch.load(checkpoint_fname)

    data = XView3DataModule(data_dir)
    test_scenes = np.array(data.get_test_scenes())

    channels = checkpoint["checkpoint_data"]["config"]["dataset"]["channels"]

    model, _ = model_from_checkpoint(checkpoint_fname, strict=True)
    box_coder = get_box_coder_from_model(model)
    model = wrap_multilabel_model_with_tta(model, tta_mode)

    normalization_op = build_normalization(checkpoint["checkpoint_data"]["config"]["normalization"])

    model = torch.jit.trace(model.cuda(), example_inputs=torch.randn(1, len(channels), 2048, 2048).cuda(), strict=False)
    tta_suffix = "" if tta_mode is None else f"_tta_{tta_mode}"

    multi_score_test_predictions = predict_multilabel_scenes(
        model=model,
        box_coder=box_coder,
        scenes=test_scenes,
        channels=channels,
        normalization=normalization_op,
        objectness_thresholds=objectness_threshold,
        vessel_thresholds=vessel_threshold,
        fishing_thresholds=fishing_threshold,
        output_predictions_dir=os.path.join(output_dir, f"test_predictions{tta_suffix}"),
        save_raw_predictions=save_predictions,
        accumulate_on_gpu=True,
        fp16=True,
        batch_size=batch_size,
    )

    multi_score_test_predictions = pd.concat(all_gather(multi_score_test_predictions)).reset_index(drop=True)

    objectness_thresholds = list(multi_score_test_predictions.objectness_threshold.unique())
    vessel_thresholds = list(multi_score_test_predictions.vessel_threshold.unique())
    fishing_thresholds = list(multi_score_test_predictions.fishing_threshold.unique())

    for objectness_threshold in objectness_thresholds:
        objectness_mask = multi_score_test_predictions.objectness_threshold == objectness_threshold

        for vessel_threshold in vessel_thresholds:
            vessel_mask = multi_score_test_predictions.vessel_threshold == vessel_threshold

            for fishing_threshold in fishing_thresholds:
                fishing_mask = multi_score_test_predictions.fishing_threshold == fishing_threshold

                test_predictions = multi_score_test_predictions[objectness_mask & vessel_mask & fishing_mask].copy().reset_index(drop=True)

                test_predictions_fname = os.path.join(
                    output_dir,
                    f"test_predictions_obj_{objectness_threshold:.2f}_vsl_{vessel_threshold:.2f}_fsh_{fishing_threshold:.2f}{tta_suffix}.csv",
                )
                test_predictions.to_csv(test_predictions_fname, index=False)


def main():
    # Give no chance to randomness
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+", help="Configuration file for inference")
    parser.add_argument("-bs", "--batch-size", type=int, default=1)

    parser.add_argument("-ot", "--objectness-threshold", nargs="+", type=float, default=0.5)
    parser.add_argument("-vt", "--vessel-threshold", nargs="+", type=float, default=0.5)
    parser.add_argument("-ft", "--fishing-threshold", nargs="+", type=float, default=0.5)

    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "-dd",
        "--data-dir",
        type=str,
        default=os.environ.get("XVIEW3_TEST_DATA_DIR", "/home/bloodaxe/develop/xview3-test"),
    )
    args = parser.parse_args()
    print("checkpoints         ", args.checkpoints)
    print("objectness_threshold", args.objectness_threshold)
    print("vessel_threshold    ", args.vessel_threshold)
    print("fishing_threshold   ", args.fishing_threshold)
    print("no_cache            ", args.no_cache)

    for checkpoint in args.checkpoints:
        for tta in [None, "flips"]:
            run_multilabel_predict(
                checkpoint,
                data_dir=args.data_dir,
                output_dir=os.path.dirname(checkpoint),
                tta_mode=tta,
                batch_size=max(1, args.batch_size // TTA_BATCH_SIZE_DIVISOR.get(tta, 1)),
                objectness_threshold=args.objectness_threshold,
                vessel_threshold=args.vessel_threshold,
                fishing_threshold=args.fishing_threshold,
                save_predictions=not args.no_cache,
            )


if __name__ == "__main__":
    main()
