import os
from pprint import pprint

import numpy as np
import pandas as pd
import torch.jit
from omegaconf import OmegaConf

from xview3 import apply_thresholds, MultilabelCircleNetCoder, CubicRootNormalization, SigmoidNormalization
from xview3.constants import IGNORE_LABEL
from xview3.inference import maybe_run_inference


# @torch.jit.optimized_execution(False)
def main(args):
    """
    Args:
        image_folder: Path to directory with all data files for inference
        scene_ids: Scene ID
        output: Path to output CSV

    Returns:

    """
    if args.scene_ids is not None:
        scene_ids = args.scene_ids.split(",")
    else:
        scene_ids = os.listdir(args.image_folder)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    box_coder = MultilabelCircleNetCoder(
        image_size=(2048, 2048),
        output_stride=2,
        max_objects=2048,
        heatmap_encoding="umich",
        ignore_value=IGNORE_LABEL,
        labels_encoding="circle",
        fixed_radius=1,
    )

    config = OmegaConf.load(args.config)
    pprint(config)

    normalization = {"vv": SigmoidNormalization(-20, 0.18), "vh": SigmoidNormalization(-20, 0.18), "bathymetry": CubicRootNormalization()}

    ensemble = torch.jit.load(config["ensemble"], map_location="cuda")
    print("Loaded ensemble from", config["ensemble"])

    all_predictions = []

    for scene_id in scene_ids:
        predictions = maybe_run_inference(
            model=ensemble,
            box_coder=box_coder,
            scene=os.path.join(args.image_folder, scene_id),
            output_predictions_dir=None,
            accumulate_on_gpu=False,
            tile_size=config["inference"]["tile_size"],
            tile_step=config["inference"]["tile_step"],
            fp16=True,
            batch_size=config["inference"]["batch_size"],
            save_raw_predictions=False,
            apply_activation=False,
            max_objects=2048,
            channels_last=config["inference"]["channels_last"],
            normalization=normalization,
            channels=["vh", "vv"],
            objectness_thresholds_lower_bound=0.3,
        )
        all_predictions.append(predictions)

    all_predictions = pd.concat(all_predictions)

    scene_predictions = apply_thresholds(
        all_predictions,
        config["thresholds"]["objectness_threshold"],
        config["thresholds"]["vessel_threshold"],
        config["thresholds"]["fishing_threshold"],
    ).drop(columns=["objectness_p", "is_vessel_p", "is_fishing_p", "objectness_threshold"])

    scene_predictions.to_csv(args.output, index=False)
    print("Saved predictions to", args.output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on xView3 reference model.")

    parser.add_argument("--image_folder", help="Path to the xView3 images")
    parser.add_argument("--scene_ids", help="Comma separated list of test scene IDs", default=None)
    parser.add_argument("--output", help="Path in which to output inference CSVs")
    parser.add_argument("--config", default="config.yaml", help="Path in inference config file")

    args = parser.parse_args()

    # Give no chance to randomness
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    main(args)
