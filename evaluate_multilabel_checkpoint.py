import argparse
import gc
import os
from datetime import timedelta
from typing import Optional

import numpy as np
import torch
from fire import Fire

from xview3 import *
from xview3.centernet.models.inference import get_box_coder_from_model
from xview3.evaluation import evaluate_on_scenes


@torch.no_grad()
def run_predict(
    checkpoint_fname: str,
    data_dir: str,
    tile_step: int,
    tta_mode: Optional[str],
    tile_size=2048,
    batch_size: int = 1,
    run_evaluation=True,
    no_model=False,
):
    checkpoint = torch.load(checkpoint_fname)
    print("Tile step", tile_step)
    data = XView3DataModule(data_dir)

    channels = checkpoint["checkpoint_data"]["config"]["dataset"]["channels"]
    _, valid_df, holdout_df, shore_root = data.train_val_split(
        splitter=checkpoint["checkpoint_data"]["config"]["dataset"]["splitter"],
        fold=checkpoint["checkpoint_data"]["config"]["dataset"]["fold"],
        num_folds=checkpoint["checkpoint_data"]["config"]["dataset"]["num_folds"],
    )
    normalization_op = build_normalization(checkpoint["checkpoint_data"]["config"]["normalization"])

    model, _ = ensemble_from_checkpoints(
        checkpoint_fnames=[checkpoint_fname],
        strict=True,
        activation="after_model",
        tta=tta_mode,
        sigmoid_outputs=[CENTERNET_OUTPUT_OBJECTNESS_MAP, CENTERNET_OUTPUT_VESSEL_MAP, CENTERNET_OUTPUT_FISHING_MAP],
        softmax_outputs=None,
        with_offset=True,
    )
    box_coder = get_box_coder_from_model(model)
    print(box_coder)

    if no_model:
        model = None
    else:
        model = model.eval().cuda()
        model = torch.jit.trace(model, example_inputs=torch.randn(1, len(channels), 2048, 2048).cuda(), strict=False)

    gc.collect()

    valid_scenes = list(valid_df.scene_path.unique())
    prefix = "valid_"
    suffix = f"_step_{tile_step}_tta_{tta_mode}"
    evaluate_on_scenes(
        model=model,
        box_coder=box_coder,
        scenes=valid_scenes,
        channels=channels,
        normalization=normalization_op,
        shore_root=shore_root,
        valid_df=valid_df,
        prefix=prefix,
        suffix=suffix,
        tile_size=tile_size,
        tile_step=tile_step,
        output_dir=os.path.join(os.path.dirname(checkpoint_fname), f"{prefix}{suffix}"),
        apply_activation=False,
        accumulate_on_gpu=True,
        batch_size=batch_size,
        fp16=True,
        run_evaluation=run_evaluation,
        save_predictions=False,
    )

    if holdout_df is not None:
        holdout_scenes = list(holdout_df.scene_path.unique())
        prefix = "holdout"
        suffix = f"_step_{tile_step}_tta_{tta_mode}"
        evaluate_on_scenes(
            model=model,
            box_coder=box_coder,
            scenes=holdout_scenes,
            channels=channels,
            normalization=normalization_op,
            shore_root=shore_root,
            tile_size=tile_size,
            tile_step=tile_step,
            valid_df=holdout_df,
            prefix=prefix,
            suffix=suffix,
            output_dir=os.path.join(os.path.dirname(checkpoint_fname), f"{prefix}{suffix}"),
            apply_activation=True,
            accumulate_on_gpu=True,
            save_predictions=False,
            batch_size=batch_size,
            fp16=True,
            run_evaluation=run_evaluation,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+", help="Configuration file for inference")
    parser.add_argument("-bs", "--batch-size", type=int, default=1)
    parser.add_argument("-tta", "--tta", type=str, default=None)
    parser.add_argument("--tile-step", default=1536, type=int)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument(
        "-dd",
        "--data-dir",
        type=str,
        default=os.environ.get("XVIEW3_DIR", "f:/datasets/xview3" if os.name == "nt" else "/home/bloodaxe/data/xview3"),
    )
    parser.add_argument("--local_rank", default=os.environ.get("LOCAL_RANK", 0), type=int)
    parser.add_argument("--world_size", default=os.environ.get("WORLD_SIZE", 1), type=int)

    args = parser.parse_args()
    world_size = args.world_size
    local_rank = args.local_rank
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(hours=4))
        torch.cuda.set_device(local_rank)
        print("Initialized distributed inference", local_rank, world_size)

    if local_rank == 0:
        print("checkpoints         ", len(args.checkpoints))
        for ck in args.checkpoints:
            print("  - ", ck)
        print("tta                 ", args.tta)
        print("no_cache            ", args.no_cache)
        print("no_model            ", args.no_model)
        print("no_eval             ", args.no_eval)

    for checkpoint in args.checkpoints:
        run_predict(
            checkpoint,
            data_dir=args.data_dir,
            tile_step=args.tile_step,
            tta_mode=args.tta,
            batch_size=args.batch_size,
            run_evaluation=not args.no_eval,
            no_model=args.no_model,
        )
        if world_size > 1:
            torch.distributed.barrier()


if __name__ == "__main__":
    # Give no chance to randomness
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    main()
