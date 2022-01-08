from typing import List, Dict
import albumentations as A
from catalyst.contrib.nn import OneCycleLRWithWarmup
from omegaconf import DictConfig
from pytorch_toolbelt.optimization.lr_schedules import (
    GradualWarmupScheduler,
    CosineAnnealingWarmRestartsWithDecay,
)
from hydra.utils import instantiate

from torch.optim.lr_scheduler import (
    CyclicLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
)

__all__ = ["get_detection_model", "get_scheduler", "build_augmentations", "build_normalization"]

from torch import nn


def get_detection_model(model_config: DictConfig) -> nn.Module:

    return instantiate(model_config, _recursive_=False)


def get_scheduler(
    optimizer,
    scheduler_name: str,
    learning_rate: float,
    num_epochs: int,
    batches_in_epoch=None,
    min_learning_rate: float = 1e-6,
    milestones=None,
    **kwargs
):
    if scheduler_name is None:
        name = ""
    else:
        name = scheduler_name.lower()

    need_warmup = "warmup_" in name
    name = name.replace("warmup_", "")

    if name == "cos":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_learning_rate, **kwargs)
    elif name == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, **kwargs)
    elif name == "cosr":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(2, num_epochs // 4), eta_min=min_learning_rate)
    elif name == "cosrd":
        scheduler = CosineAnnealingWarmRestartsWithDecay(optimizer, T_0=max(2, num_epochs // 6), eta_min=min_learning_rate)
    elif name in {"1cycle", "one_cycle"}:
        scheduler = OneCycleLRWithWarmup(
            optimizer,
            lr_range=(learning_rate, min_learning_rate),
            num_steps=batches_in_epoch * num_epochs,
            **kwargs,
        )
    elif name == "exp":
        scheduler = ExponentialLR(optimizer, **kwargs)
    elif name == "clr":
        scheduler = CyclicLR(
            optimizer,
            base_lr=min_learning_rate,
            max_lr=learning_rate,
            step_size_up=batches_in_epoch // 4,
            # mode='exp_range',
            gamma=0.99,
        )
    elif name == "multistep":
        milestones = [int(num_epochs * m) for m in milestones]
        scheduler = MultiStepLR(optimizer, milestones=milestones, **kwargs)
    elif name == "simple":
        scheduler = MultiStepLR(optimizer, milestones=[int(num_epochs * 0.5), int(num_epochs * 0.8)], **kwargs)
    else:
        raise KeyError(f"Unsupported scheduler name {name}")

    if need_warmup:
        scheduler = GradualWarmupScheduler(optimizer, 1.0, 5, after_scheduler=scheduler)
        print("Adding warmup")

    return scheduler


def demangle_albumentations(config):
    if str.startswith(config["_target_"], "A."):
        config["_target_"] = config["_target_"].replace("A.", "albumentations.")
    return config


def build_augmentations(config: DictConfig) -> List[A.BasicTransform]:
    augs = []
    if config is not None:
        for section in config:
            section = demangle_albumentations(section)
            aug = instantiate(section)
            augs.append(aug)
    return augs


def build_normalization(config: DictConfig) -> Dict[str, A.ImageOnlyTransform]:
    augs = {}
    if config is not None:
        for channel, channel_config in config["channels"].items():
            if channel_config is not None:
                channel_config = demangle_albumentations(channel_config)
                aug = instantiate(channel_config)
                augs[channel] = aug
    return augs
