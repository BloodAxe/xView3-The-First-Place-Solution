import collections
import gc
import os
from typing import List, Optional, Union, Tuple, Dict, Any

import albumentations
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from pytorch_toolbelt.inference import (
    ApplySigmoidTo,
    ApplySoftmaxTo,
    Ensembler,
    GeneralizedTTA,
    MultiscaleTTA,
    d2_image_augment,
    d4_image_augment,
    d4_image_deaugment,
    d2_image_deaugment,
    flips_image_deaugment,
    flips_image_augment,
    fliplr_image_augment,
    fliplr_image_deaugment,
)
from pytorch_toolbelt.utils import to_numpy, fs
from torch import nn
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from pytorch_toolbelt.utils.distributed import is_main_process, get_rank, get_world_size, all_gather

from xview3.centernet.bboxer import (
    MultilabelCircleNetDecodeResult,
    MultilabelCircleNetCoder,
)
from xview3.centernet.constants import (
    CENTERNET_OUTPUT_SIZE,
    CENTERNET_OUTPUT_OFFSET,
    CENTERNET_OUTPUT_OBJECTNESS_MAP,
    CENTERNET_OUTPUT_VESSEL_MAP,
    CENTERNET_OUTPUT_FISHING_MAP,
)
from xview3.centernet.models.inference import (
    multilabel_centernet_tiled_inference,
    get_box_coder_from_model,
)
from xview3.dataset import (
    XView3DataModule,
    read_multichannel_image,
    stack_multichannel_image,
)

__all__ = [
    "average_checkpoints",
    "ensemble_from_checkpoints",
    "model_from_checkpoint",
    "ensemble_from_config",
    "wrap_multilabel_model_with_tta",
    "maybe_run_inference",
]


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights. Original implementation taken from:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16
    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location="cpu",
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["model_state_dict"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError("For checkpoint {}, expected list of params: {}, " "but found: {}".format(f, params_keys, model_params_keys))
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model_state_dict"] = averaged_params
    return new_state


def model_from_checkpoint(checkpoint_config: Union[str, Dict], **kwargs) -> Tuple[nn.Module, Dict]:
    if isinstance(checkpoint_config, collections.Mapping):
        if "average_checkpoints" in checkpoint_config:
            checkpoint = average_checkpoints(checkpoint_config["average_checkpoints"])
        else:
            checkpoint_name = checkpoint_config["checkpoint"]
            if os.path.isfile(checkpoint_name):
                checkpoint = torch.load(checkpoint_name, map_location="cpu")
            else:
                checkpoint = torch.hub.load_state_dict_from_url(checkpoint_name)

        model_config = checkpoint["checkpoint_data"]["config"]["model"]
    else:
        checkpoint_name = checkpoint_config

        if os.path.isfile(checkpoint_name):
            checkpoint = torch.load(checkpoint_name, map_location="cpu")
        else:
            checkpoint = torch.hub.load_state_dict_from_url(checkpoint_name)

        model_config = checkpoint["checkpoint_data"]["config"]["model"]

    model_state_dict = checkpoint["model_state_dict"]

    model = instantiate(model_config, _recursive_=False)
    model.load_state_dict(model_state_dict, strict=False)

    return model.eval(), checkpoint


def wrap_multilabel_model_with_tta(model, tta_mode, with_offset=True, size_offsets=(0, -32, -64, +32, +64)):
    from xview3.centernet import (
        CENTERNET_OUTPUT_VESSEL_MAP,
        CENTERNET_OUTPUT_FISHING_MAP,
        CENTERNET_OUTPUT_OBJECTNESS_MAP,
        CENTERNET_OUTPUT_OFFSET,
        CENTERNET_OUTPUT_SIZE,
    )

    keys_to_deaug = [
        CENTERNET_OUTPUT_VESSEL_MAP,
        CENTERNET_OUTPUT_FISHING_MAP,
        CENTERNET_OUTPUT_OBJECTNESS_MAP,
        CENTERNET_OUTPUT_SIZE,
    ]
    if with_offset:
        keys_to_deaug.append(CENTERNET_OUTPUT_OFFSET)

    def _make_deaug_dict(keys, fn):
        return dict((key, fn) for key in keys)

    if tta_mode == "d4":
        return GeneralizedTTA(model, augment_fn=d4_image_augment, deaugment_fn=_make_deaug_dict(keys_to_deaug, d4_image_deaugment))
    elif tta_mode == "ms":
        return MultiscaleTTA(model, size_offsets)
    elif tta_mode == "d2-ms":
        return MultiscaleTTA(GeneralizedTTA(model, d2_image_augment, d2_image_deaugment), size_offsets)
    elif tta_mode == "d2":
        model = GeneralizedTTA(model, d2_image_augment, deaugment_fn=_make_deaug_dict(keys_to_deaug, d2_image_deaugment))
    elif tta_mode == "flips":
        model = GeneralizedTTA(model, flips_image_augment, deaugment_fn=_make_deaug_dict(keys_to_deaug, flips_image_deaugment))
    elif tta_mode == "fliplr":
        model = GeneralizedTTA(model, fliplr_image_augment, deaugment_fn=_make_deaug_dict(keys_to_deaug, fliplr_image_deaugment))
    elif tta_mode is None:
        return model
    else:
        raise KeyError("Unusupported TTA mode '" + tta_mode + "'")

    return model


def ensemble_from_checkpoints(
    checkpoint_fnames: List[str],
    strict=True,
    sigmoid_outputs=None,
    softmax_outputs=None,
    activation: str = "after_model",
    tta: Optional[str] = None,
    with_offset=True,
):
    if activation not in {None, "None", "after_model", "after_tta", "after_ensemble"}:
        raise KeyError(activation)

    models = []
    checkpoints = []

    for ck in checkpoint_fnames:
        model, checkpoint = model_from_checkpoint(ck, strict=strict)
        models.append(model)
        checkpoints.append(checkpoint)

    if activation == "after_model":
        if sigmoid_outputs is not None:
            models = [ApplySigmoidTo(m, output_key=sigmoid_outputs) for m in models]
            print("Applying sigmoid activation to", sigmoid_outputs, "after each model", len(models))

        if softmax_outputs is not None:
            models = [ApplySoftmaxTo(m, output_key=softmax_outputs) for m in models]
            print("Applying softmax activation to", softmax_outputs, "after each model", len(models))

    if len(models) > 1:
        model = Ensembler(models)
        if activation == "after_ensemble":
            if sigmoid_outputs is not None:
                model = ApplySigmoidTo(model, output_key=sigmoid_outputs)
                print("Applying sigmoid activation to", sigmoid_outputs, "after ensemble")

            if softmax_outputs is not None:
                model = ApplySoftmaxTo(model, output_key=softmax_outputs)
                print("Applying softmax activation to", softmax_outputs, "after ensemble")

    else:
        assert len(models) == 1
        model = models[0]

    if tta not in {None, "None"}:
        model = wrap_multilabel_model_with_tta(model, tta, with_offset=with_offset)
        print("Wrapping models with TTA", tta)

    if activation == "after_tta":
        if sigmoid_outputs is not None:
            model = ApplySigmoidTo(model, output_key=sigmoid_outputs)
            print("Applying sigmoid activation to ", sigmoid_outputs, " after TTA")
        if softmax_outputs is not None:
            model = ApplySoftmaxTo(model, output_key=softmax_outputs)
            print("Applying softmax activation to", softmax_outputs, "after TTA")

    return model.eval(), checkpoints


def ensemble_from_config(config: Dict[str, Any]):
    model, checkpoints = ensemble_from_checkpoints(
        checkpoint_fnames=config["ensemble"]["models"],
        strict=True,
        activation=config["ensemble"]["activation_after"],
        tta=config["ensemble"]["tta"],
        sigmoid_outputs=config["ensemble"]["sigmoid_outputs"],
        softmax_outputs=config["ensemble"]["softmax_outputs"],
        with_offset=config["ensemble"]["with_offset"],
    )
    box_coder = get_box_coder_from_model(model)
    model = model.eval().cuda()
    return model, checkpoints, box_coder


@torch.jit.optimized_execution(False)
def predict_multilabel_scenes(
    model,
    box_coder: MultilabelCircleNetCoder,
    scenes: List[str],
    channels: List[str],
    tile_step: int,
    tile_size: int,
    objectness_thresholds_lower_bound: float,
    normalization: Dict[str, albumentations.ImageOnlyTransform],
    accumulate_on_gpu: bool,
    fp16: bool,
    batch_size: int,
    apply_activation: bool,
    save_raw_predictions: bool,
    max_objects: int,
    channels_last: bool,
    output_predictions_dir=None,
) -> pd.DataFrame:
    if output_predictions_dir is not None:
        os.makedirs(output_predictions_dir, exist_ok=True)

    all_predictions = []

    scenes = np.array(scenes)
    world_size, local_rank = get_world_size(), get_rank()

    if world_size > 1:
        sampler = DistributedSampler(scenes, world_size, local_rank, shuffle=False)
        rank_local_indexes = np.array(list(iter(sampler)))
        scenes = scenes[rank_local_indexes]
        print("Node", local_rank, "got", len(scenes), "to process")
        torch.distributed.barrier()

    for scene in tqdm(scenes, desc=f"Inference at Node {local_rank}/{world_size}", position=local_rank):
        gc.collect()
        scene_id = fs.id_from_fname(scene)
        predictions = maybe_run_inference(
            model=model,
            box_coder=box_coder,
            scene=scene,
            output_predictions_dir=output_predictions_dir,
            accumulate_on_gpu=accumulate_on_gpu,
            tile_size=tile_size,
            tile_step=tile_step,
            fp16=fp16,
            batch_size=batch_size,
            save_raw_predictions=save_raw_predictions,
            apply_activation=apply_activation,
            max_objects=max_objects,
            channels_last=channels_last,
            normalization=normalization,
            channels=channels,
            objectness_thresholds_lower_bound=objectness_thresholds_lower_bound,
        )

        all_predictions.append(predictions)
        if output_predictions_dir is not None:
            predictions.to_csv(os.path.join(output_predictions_dir, scene_id + ".csv"), index=False)

    all_predictions = pd.concat(all_predictions).reset_index(drop=True)

    if world_size > 1:
        torch.distributed.barrier()
        all_predictions = pd.concat(all_gather(all_predictions)).reset_index(drop=True)

    return all_predictions


def maybe_run_inference(
    model,
    box_coder,
    scene,
    output_predictions_dir,
    channels,
    normalization,
    objectness_thresholds_lower_bound: float,
    tile_size,
    tile_step,
    accumulate_on_gpu,
    fp16,
    batch_size,
    save_raw_predictions,
    apply_activation,
    max_objects,
    channels_last,
):
    scene_id = fs.id_from_fname(scene)
    predictions_computed_offline = False

    if output_predictions_dir is not None:
        raw_predictions_file = os.path.join(output_predictions_dir, scene_id + ".npz")
        decoded_predictions_file = os.path.join(output_predictions_dir, scene_id + ".csv")

        if os.path.isfile(decoded_predictions_file):
            try:
                predictions = pd.read_csv(decoded_predictions_file)
                return predictions
            except Exception as e:
                print(e)
                predictions_computed_offline = False
        elif os.path.isfile(raw_predictions_file):
            try:
                saved_predictions = np.load(raw_predictions_file, allow_pickle=True)
                outputs = dict(
                    CENTERNET_OUTPUT_OBJECTNESS_MAP=torch.from_numpy(saved_predictions[CENTERNET_OUTPUT_OBJECTNESS_MAP]),
                    CENTERNET_OUTPUT_VESSEL_MAP=torch.from_numpy(saved_predictions[CENTERNET_OUTPUT_VESSEL_MAP]),
                    CENTERNET_OUTPUT_FISHING_MAP=torch.from_numpy(saved_predictions[CENTERNET_OUTPUT_FISHING_MAP]),
                    CENTERNET_OUTPUT_SIZE=torch.from_numpy(saved_predictions[CENTERNET_OUTPUT_SIZE]),
                    CENTERNET_OUTPUT_OFFSET=torch.from_numpy(saved_predictions[CENTERNET_OUTPUT_OFFSET])
                    if CENTERNET_OUTPUT_OFFSET in saved_predictions
                    else None,
                )
                predictions_computed_offline = True
            except Exception as e:
                print(e)
                predictions_computed_offline = False

    if not predictions_computed_offline:
        image = read_multichannel_image(scene, channels)
        for channel_name in set(channels):
            image[channel_name] = normalization[channel_name](image=image[channel_name])["image"]
        image = stack_multichannel_image(image, channels)

        outputs = multilabel_centernet_tiled_inference(
            model,
            image,
            box_coder=box_coder,
            tile_size=tile_size,
            tile_step=tile_step,
            accumulate_on_gpu=accumulate_on_gpu,
            fp16=fp16,
            batch_size=batch_size,
            channels_last=channels_last,
        )

        if save_raw_predictions and output_predictions_dir is not None:
            raw_predictions_file = os.path.join(output_predictions_dir, scene_id + ".npz")
            predictions_dict = dict(
                CENTERNET_OUTPUT_OBJECTNESS_MAP=to_numpy(outputs[CENTERNET_OUTPUT_OBJECTNESS_MAP]),
                CENTERNET_OUTPUT_VESSEL_MAP=to_numpy(outputs[CENTERNET_OUTPUT_VESSEL_MAP]),
                CENTERNET_OUTPUT_FISHING_MAP=to_numpy(outputs[CENTERNET_OUTPUT_FISHING_MAP]),
                CENTERNET_OUTPUT_SIZE=to_numpy(outputs[CENTERNET_OUTPUT_SIZE]),
            )
            if CENTERNET_OUTPUT_OFFSET in outputs:
                predictions_dict[CENTERNET_OUTPUT_OFFSET] = to_numpy(outputs[CENTERNET_OUTPUT_OFFSET])
            np.savez(raw_predictions_file, **predictions_dict)

    preds: MultilabelCircleNetDecodeResult = box_coder.decode(
        objectness_map=outputs[CENTERNET_OUTPUT_OBJECTNESS_MAP],
        is_vessel_map=outputs[CENTERNET_OUTPUT_VESSEL_MAP],
        is_fishing_map=outputs[CENTERNET_OUTPUT_FISHING_MAP],
        length_map=outputs[CENTERNET_OUTPUT_SIZE],
        offset_map=outputs.get(CENTERNET_OUTPUT_OFFSET, None),
        apply_activation=apply_activation,
        max_objects=max_objects,
    )

    pos_mask = preds.scores[0] >= objectness_thresholds_lower_bound

    centers = to_numpy(preds.centers[0][pos_mask]).astype(int)
    scores = to_numpy(preds.scores[0, pos_mask]).astype(np.float32)
    lengths = XView3DataModule.decode_lengths(preds.lengths[0, pos_mask])

    is_vessel_prob = to_numpy(preds.is_vessel[0, pos_mask]).astype(np.float32)
    is_fishing_prob = to_numpy(preds.is_fishing[0, pos_mask]).astype(np.float32)

    predictions = collections.defaultdict(list)
    for (
        (detect_scene_column, detect_scene_row),
        objectness_score,
        is_vessel_p,
        is_fishing_p,
        vessel_length_m,
    ) in zip(centers, scores, is_vessel_prob, is_fishing_prob, lengths):
        predictions["vessel_length_m"].append(vessel_length_m)
        predictions["detect_scene_row"].append(detect_scene_row)
        predictions["detect_scene_column"].append(detect_scene_column)
        predictions["scene_id"].append(scene_id)
        # Scores
        predictions["objectness_p"].append(objectness_score)
        predictions["is_vessel_p"].append(is_vessel_p)
        predictions["is_fishing_p"].append(is_fishing_p)
        # Thresholds
        predictions["objectness_threshold"].append(objectness_thresholds_lower_bound)

    predictions = pd.DataFrame.from_dict(predictions)
    return predictions
