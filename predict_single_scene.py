import gc
import os
from datetime import timedelta
from typing import Dict, Any, List

import cv2
import numpy as np
import torch
import torch.distributed as dist
from fire import Fire
from omegaconf import OmegaConf
from pytorch_toolbelt.utils.distributed import is_main_process
from pytorch_toolbelt.utils import fs
from tqdm import tqdm

from xview3 import *
from xview3.centernet.visualization import create_false_color_composite, vis_detections_opencv
from xview3.constants import PIX_TO_M
from xview3.inference import (
    predict_multilabel_scenes,
)


def run_multilabel_predict(config: Dict[str, Any], scenes: List[str], submission_dir: str):
    model, checkpoints, box_coder = ensemble_from_config(config)

    checkpoint = checkpoints[0]
    normalization_op = build_normalization(checkpoint["checkpoint_data"]["config"]["normalization"])
    channels = checkpoint["checkpoint_data"]["config"]["dataset"]["channels"]

    channels_last = config["inference"]["channels_last"]
    tile_size = config["inference"]["tile_size"]
    tile_step = config["inference"]["tile_step"]

    os.makedirs(submission_dir, exist_ok=True)

    if config["inference"]["use_traced_model"]:
        traced_model_path = os.path.join(submission_dir, "traced_ensemble.jit")
        if os.path.exists(traced_model_path):
            model = torch.jit.load(traced_model_path)
        else:
            with torch.no_grad():
                if channels_last:
                    model = model.to(memory_format=torch.channels_last)
                    print("Using channels last format")

                model = torch.jit.trace(
                    model,
                    example_inputs=torch.randn(1, len(channels), tile_size, tile_size).cuda(),
                    strict=False,
                )
                # if is_main_process():
                #     torch.jit.save(model, traced_model_path)

    del checkpoints
    gc.collect()

    os.makedirs(submission_dir, exist_ok=True)

    multi_score_test_predictions = predict_multilabel_scenes(
        model=model,
        box_coder=box_coder,
        scenes=scenes,
        channels=channels,
        normalization=normalization_op,
        output_predictions_dir=submission_dir,
        save_raw_predictions=False,
        apply_activation=False,
        # Inference options
        accumulate_on_gpu=config["inference"]["accumulate_on_gpu"],
        tile_size=tile_size,
        tile_step=tile_step,
        batch_size=config["inference"]["batch_size"],
        fp16=config["inference"]["fp16"],
        channels_last=channels_last,
        # Thresholds
        objectness_thresholds_lower_bound=0.3,
        max_objects=2048,
    )

    if is_main_process():
        multi_score_test_predictions.to_csv(os.path.join(submission_dir, "unfiltered_predictions.csv"), index=False)

        for thresholds in config["thresholds"]:
            objectness_threshold = float(thresholds["objectness"])
            vessel_threshold = float(thresholds["is_vessel"])
            fishing_threshold = float(thresholds["is_fishing"])

            test_predictions = apply_thresholds(multi_score_test_predictions, objectness_threshold, vessel_threshold, fishing_threshold)

            test_predictions_fname = os.path.join(
                submission_dir,
                f"predictions_obj_{objectness_threshold:.3f}_vsl_{vessel_threshold:.3f}_fsh_{fishing_threshold:.3f}.csv",
            )
            test_predictions.to_csv(test_predictions_fname, index=False)


        if True:
            for scene_path in tqdm(scenes, desc="Making visualizations"):
                scene_id = fs.id_from_fname(scene_path)
                scene_df = test_predictions[test_predictions.scene_id == scene_id]

                image = read_multichannel_image(scene_path, ["vv", "vh"])

                normalize = SigmoidNormalization()
                size_down_4 = image["vv"].shape[1] // 4, image["vv"].shape[0] // 4
                image_rgb = create_false_color_composite(
                    normalize(image=cv2.resize(image["vv"], dsize=size_down_4, interpolation=cv2.INTER_AREA))["image"],
                    normalize(image=cv2.resize(image["vh"], dsize=size_down_4, interpolation=cv2.INTER_AREA))["image"],
                )
                image_rgb[~np.isfinite(image_rgb)] = 0

                targets = XView3DataModule.get_multilabel_targets_from_df(scene_df)
                centers = (targets.centers * 0.25).astype(int)
                image_rgb = vis_detections_opencv(
                    image_rgb,
                    centers=centers,
                    lengths=XView3DataModule.decode_lengths(targets.lengths) / PIX_TO_M,
                    is_vessel_vec=targets.is_vessel,
                    is_fishing_vec=targets.is_fishing,
                    is_vessel_probs=None,
                    is_fishing_probs=None,
                    scores=targets.objectness_probs,
                    show_title=True,
                    alpha=0.1,
                )
                cv2.imwrite(os.path.join(submission_dir, scene_id + ".jpg"), image_rgb)


def main(
    *images: List[str],
    config: str = None,
    output_dir: str = None,
    local_rank=int(os.environ.get("LOCAL_RANK", 0)),
    world_size=int(os.environ.get("WORLD_SIZE", 1))
):
    if config is None:
        raise ValueError("--config must be set")
    if output_dir is None:
        raise ValueError("--output_dir must be set")

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(hours=4))
        torch.cuda.set_device(local_rank)
        print("Initialized distributed inference", local_rank, world_size)

    run_multilabel_predict(OmegaConf.load(config), scenes=images, submission_dir=output_dir)

    if world_size > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    # Give no chance to randomness
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    Fire(main)
