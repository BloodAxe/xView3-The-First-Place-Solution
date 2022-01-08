import gc
import os
from datetime import timedelta
from typing import Any, Dict

import numpy as np
import torch
from fire import Fire
from omegaconf import OmegaConf

from pytorch_toolbelt.utils.distributed import is_main_process

from xview3 import *


def evaluate_ensemble_on_holdout(config: Dict[str, Any], data_dir: str):
    data = XView3DataModule(data_dir)

    model, checkpoints, box_coder = ensemble_from_config(config)
    print(box_coder)

    checkpoint = checkpoints[0]
    normalization_op = build_normalization(checkpoint["checkpoint_data"]["config"]["normalization"])
    channels = checkpoint["checkpoint_data"]["config"]["dataset"]["channels"]
    _, _, holdout_df, shore_root = data.train_val_split(
        splitter=checkpoint["checkpoint_data"]["config"]["dataset"]["splitter"],
        fold=checkpoint["checkpoint_data"]["config"]["dataset"]["fold"],
        num_folds=checkpoint["checkpoint_data"]["config"]["dataset"]["num_folds"],
    )
    holdout_scenes = list(holdout_df.scene_path.unique())
    if is_main_process():
        print("Holdout scenes", len(holdout_scenes))

    channels_last = config["inference"]["channels_last"]
    tile_size = config["inference"]["tile_size"]
    tile_step = config["inference"]["tile_step"]
    tta_mode = config["ensemble"]["tta"]

    submission_dir = config["submission_dir"]
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
                if is_main_process():
                    torch.jit.save(model, traced_model_path)

    del checkpoints
    gc.collect()
    prefix = "holdout_"
    suffix = f"_step_{tile_step}_tta_{tta_mode}"
    evaluate_on_scenes(
        model=model,
        box_coder=box_coder,
        scenes=holdout_scenes,
        channels=channels,
        normalization=normalization_op,
        shore_root=shore_root,
        valid_df=holdout_df,
        prefix=prefix,
        suffix=suffix,
        output_dir=os.path.join(submission_dir, f"{prefix}{suffix}"),
        apply_activation=False,
        # Inference options
        accumulate_on_gpu=config["inference"]["accumulate_on_gpu"],
        tile_size=tile_size,
        tile_step=tile_step,
        batch_size=config["inference"]["batch_size"],
        fp16=config["inference"]["fp16"],
        channels_last=channels_last,
        # Thresholds
        save_predictions=False,
        objectness_thresholds=config["evaluation"]["objectness_thresholds"],
    )


def main(
    *configs,
    data_dir=os.environ.get("XVIEW3_DIR", "f:/datasets/xview3" if os.name == "nt" else "/home/bloodaxe/data/xview3"),
    local_rank=int(os.environ.get("LOCAL_RANK", 0)),
    world_size=int(os.environ.get("WORLD_SIZE", 1))
):
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(hours=4))
        torch.cuda.set_device(local_rank)
        print("Initialized distributed inference", local_rank, world_size)

    for config in configs:
        evaluate_ensemble_on_holdout(OmegaConf.load(config), data_dir=data_dir)

        if world_size > 1:
            torch.distributed.barrier()


if __name__ == "__main__":
    # Give no chance to randomness
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    Fire(main)
