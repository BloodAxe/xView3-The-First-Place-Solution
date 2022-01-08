import collections
import os
import traceback
from datetime import datetime
from time import sleep
from typing import Tuple, Any, Optional, Dict, List, OrderedDict

import hydra.utils
import torch
from catalyst.callbacks import (
    MetricAggregationCallback,
    CriterionCallback,
    SchedulerCallback,
    AMPOptimizerCallback,
    OptimizerCallback,
    TimerCallback,
)
from catalyst.contrib.nn import OneCycleLRWithWarmup
from catalyst.core import Callback
from catalyst.data import DistributedSamplerWrapper
from catalyst.runners import SupervisedRunner
from catalyst.utils import unpack_checkpoint, load_checkpoint
from omegaconf import OmegaConf, DictConfig
from pytorch_toolbelt.datasets import *
from pytorch_toolbelt.optimization.functional import freeze_model, get_lr_decay_parameters, get_optimizable_parameters
from pytorch_toolbelt.utils import fs, transfer_weights, count_parameters
from pytorch_toolbelt.utils.catalyst import (
    report_checkpoint,
    MixupCriterionCallback,
    MixupInputCallback,
)
from pytorch_toolbelt.utils.catalyst.pipeline import (
    get_optimizer_cls,
    scale_learning_rate_for_ddp,
)
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DistributedSampler, Sampler, DataLoader

from .dataset import *
from .factory import get_scheduler
from .version import *

__all__ = ["Pipeline"]

_GLOBAL_NCCL_INITIALIZED_STATUS = False


class Pipeline:
    _distributed = False
    _is_master = True

    def __init__(self, cfg):
        self.cfg = cfg
        self.discover_distributed_params()
        self.experiment_dir = None

    def discover_distributed_params(self):
        global _GLOBAL_NCCL_INITIALIZED_STATUS

        if _GLOBAL_NCCL_INITIALIZED_STATUS:
            self.master_print("In seems NCCL has been already initialized. Skipping second initialization")
            return

        self._distributed = int(self.cfg.world_size) > 1
        if not self._distributed:
            return
        sleep(int(self.cfg.world_size) * 0.1)
        print("Initializing init_process_group", self.cfg.local_rank, flush=True)
        torch.cuda.set_device(int(self.cfg.local_rank))
        torch.distributed.init_process_group(backend="nccl")

        sleep(int(self.cfg.world_size) * 0.1)
        print("Initialized init_process_group", int(self.cfg.local_rank), flush=True)

        self._is_master = (int(self.cfg.local_rank) == 0) | (not self._distributed)
        _GLOBAL_NCCL_INITIALIZED_STATUS = True

    @property
    def distributed(self):
        return self._distributed

    @property
    def is_master(self):
        return self._is_master

    def master_print(self, *args, **kwargs):
        if self.is_master:
            print(*args, **kwargs)

    def get_criterion_callback(
        self,
        loss_config,
        target_key,
        output_key,
        prefix: str,
        loss_weight: float = 1.0,
        mixup: bool = False,
        train_only: bool = False,
    ) -> Tuple[Dict, CriterionCallback, str]:
        if target_key is not None and not isinstance(target_key, str):
            target_key = OmegaConf.to_container(target_key)
        if output_key is not None and not isinstance(output_key, str):
            output_key = OmegaConf.to_container(output_key)

        criterion = self.get_loss(loss_config)
        if not isinstance(criterion, nn.Module):
            raise RuntimeError("Loss module must be subclass of nn.Module")
        criterions_dict = {f"{prefix}": criterion}

        if mixup:
            criterion_callback = MixupCriterionCallback(
                prefix=f"{prefix}",
                input_key=target_key,
                output_key=output_key,
                criterion_key=f"{prefix}",
                multiplier=float(loss_weight),
            )
        elif train_only:
            criterion_callback = TrainOnlyCriterionCallback(
                prefix=f"{prefix}",
                input_key=target_key,
                output_key=output_key,
                criterion_key=f"{prefix}",
                multiplier=float(loss_weight),
            )
        else:
            criterion_callback = CriterionCallback(
                prefix=f"{prefix}",
                input_key=target_key,
                output_key=output_key,
                criterion_key=f"{prefix}",
                multiplier=float(loss_weight),
            )

        return criterions_dict, criterion_callback, prefix

    def build_criterions(self, config: Dict) -> Tuple[Dict[str, nn.Module], List[Callback]]:
        losses = []
        criterions_dict = {}
        callbacks = []

        mixup = config["train"].get("mixup", False)
        losses_aggregation = config["loss"]["aggregation"]
        criterions_config: List[Dict] = config["loss"]["losses"]

        if mixup:
            mixup_a = self.cfg["train"].get("mixup_a", 0.5)
            mixup_p = self.cfg["train"].get("mixup_p", 0.5)
            callbacks.append(
                MixupInputCallback(
                    fields=[INPUT_IMAGE_KEY],
                    alpha=mixup_a,
                    p=mixup_p,
                )
            )
            self.master_print("Using Mixup", "alpha", mixup_a, "p", mixup_p)

        self.master_print("Losses")
        train_only = config["loss"].get("train_only", False)
        for criterion_cfg in criterions_config:
            loss_weight = criterion_cfg.get("weight", 1.0)

            criterion_loss, criterion_callback, criterion_name = self.get_criterion_callback(
                criterion_cfg["loss"],
                prefix="losses/" + criterion_cfg["prefix"],
                target_key=criterion_cfg["target_key"],
                output_key=criterion_cfg["output_key"],
                loss_weight=float(loss_weight),
                mixup=mixup,
                train_only=train_only,
            )
            criterions_dict.update(criterion_loss)
            callbacks.append(criterion_callback)
            losses.append(criterion_name)

            self.master_print(
                "  ",
                criterion_name,
                criterion_loss[criterion_name].__class__.__name__,
                "weight",
                loss_weight,
            )
            self.master_print(
                "    ",
                "target",
                criterion_cfg["target_key"],
            )
            self.master_print(
                "    ",
                "output",
                criterion_cfg["output_key"],
            )

        callbacks.append(
            MetricAggregationCallback(prefix="loss", metrics=config["loss"].get("losses_weights", losses), mode=losses_aggregation)
        )
        return criterions_dict, callbacks

    def get_loss(self, loss_config: Dict):
        return hydra.utils.call(loss_config, **{"_convert_": "all"})

    def build_datasets(self, config) -> Tuple[KeypointsDataset, KeypointsDataset, Optional[Sampler], List[Callback]]:
        raise NotImplemented

    def build_metrics(self, config, loaders, model) -> List[Callback]:
        return []

    def build_loaders(self, config: DictConfig) -> Tuple[collections.OrderedDict, List[Callback]]:
        train_ds, valid_ds, train_sampler, dataset_callbacks = self.build_datasets(config)
        valid_sampler = None

        if self.distributed:
            world_size = torch.distributed.get_world_size()
            local_rank = torch.distributed.get_rank()
            if train_sampler is not None:
                train_sampler = DistributedSamplerWrapper(train_sampler, world_size, local_rank, shuffle=True)
            else:
                train_sampler = DistributedSampler(train_ds, world_size, local_rank, shuffle=True)
            valid_sampler = DistributedSampler(valid_ds, world_size, local_rank, shuffle=False)

        loaders = collections.OrderedDict()

        loaders["train"] = DataLoader(
            train_ds,
            batch_size=config.train.loaders.train.batch_size,
            num_workers=config.train.loaders.train.num_workers,
            pin_memory=False,
            drop_last=True,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=train_ds.get_collate_fn(),
        )
        self.master_print("Train loader")
        self.master_print("  Length    ", len(loaders["train"]))
        self.master_print("  Batch Size", config.train.loaders.train.batch_size)
        self.master_print("  Workers   ", config.train.loaders.train.num_workers)

        loaders["valid"] = DataLoader(
            valid_ds,
            batch_size=config.train.loaders.valid.batch_size,
            num_workers=config.train.loaders.valid.num_workers,
            pin_memory=False,
            sampler=valid_sampler,
            drop_last=False,
            collate_fn=valid_ds.get_collate_fn(),
        )
        self.master_print("Valid loader")
        self.master_print("  Length    ", len(loaders["valid"]))
        self.master_print("  Batch Size", config.train.loaders.valid.batch_size)
        self.master_print("  Workers   ", config.train.loaders.valid.num_workers)

        return loaders, dataset_callbacks

    def build_distributed_params(self, config: DictConfig) -> Dict:
        fp16 = config.optimizer.get("fp16", False)

        if self.distributed:
            local_rank = torch.distributed.get_rank()
            distributed_params = {"rank": local_rank, "syncbn": True}
            if fp16:
                distributed_params["amp"] = True
            if self.cfg.find_unused:
                distributed_params["find_unused_parameters"] = True
        else:
            if fp16:
                distributed_params = {}
                distributed_params["amp"] = True
            else:
                distributed_params = False

        return distributed_params

    def build_optimizer(self, model: nn.Module, loaders: OrderedDict) -> Tuple[Optimizer, Any, List[Callback]]:
        optimizer_config = self.cfg["optimizer"]
        optimizer_use_fp16 = optimizer_config.get("fp16", False)

        optimizer_name = str(optimizer_config["name"])

        wd_on_bias = bool(optimizer_config.get("wd_on_bias", True))
        accumulation_steps = int(optimizer_config.get("accumulation", 1))
        optimizer_params = optimizer_config["params"]
        optimizer_params = scale_learning_rate_for_ddp(optimizer_params)

        optimizer = self.get_optimizer(
            model=model,
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
            apply_weight_decay_to_bias=wd_on_bias,
            layerwise_params=optimizer_config.get("layerwise_params", None),
        )

        if optimizer_use_fp16:
            opt_callback = AMPOptimizerCallback(accumulation_steps=accumulation_steps)
        else:
            opt_callback = OptimizerCallback(accumulation_steps=accumulation_steps, decouple_weight_decay=False)

        scheduler_params = self.cfg["scheduler"]
        num_epochs = self.cfg["train"]["epochs"]

        scheduler_name = str(scheduler_params["scheduler_name"])
        scheduler = get_scheduler(
            optimizer,
            learning_rate=optimizer_params["lr"],
            num_epochs=num_epochs,
            batches_in_epoch=len(loaders["train"]),
            **scheduler_params,
        )

        callbacks = [opt_callback]
        if isinstance(scheduler, (CyclicLR, OneCycleLRWithWarmup)):
            callbacks += [SchedulerCallback(mode="batch")]
        else:
            callbacks += [SchedulerCallback(mode="epoch")]

        self.master_print("Optimizer        :", optimizer_name)
        self.master_print("Model            :", self.cfg.model.config.slug)
        self.master_print(
            "  Parameters     :",
            count_parameters(
                model, ["backbone", "rpn", "roi_heads", "encoder", "decoder", "head", "fuse", "extra_stages", "center", "mask"]
            ),
        )
        self.master_print("  FP16           :", optimizer_use_fp16)
        self.master_print("  Learning rate  :", optimizer_params["lr"])
        self.master_print("  Weight decay   :", optimizer_params.get("weight_decay", 0))
        self.master_print("  WD on bias     :", wd_on_bias)
        self.master_print("  Accumulation   :", accumulation_steps)
        self.master_print("Params           :")
        for k, v in optimizer_params.items():
            self.master_print(f"  {k}:", v)
        self.master_print("Scheduler        :", scheduler_name)

        return optimizer, scheduler, callbacks

    def get_optimizer(
        self,
        model: nn.Module,
        optimizer_name: str,
        optimizer_params: Dict[str, Any],
        apply_weight_decay_to_bias: bool = True,
        layerwise_params=None,
    ) -> Optimizer:
        """
        Construct an Optimizer for given model
        Args:
            model: Model to optimize. Only parameters that require_grad will be used
            optimizer_name: Name of the optimizer (case-insensitive). Supports native pytorch optimizers, apex and
                optimizers from pytorch-optimizers package.
            optimizer_params: Dict of optimizer params (lr, weight_decay, eps, etc)
            apply_weight_decay_to_bias: Whether to apply weight decay on bias parameters. Default is True
        Returns:
            Optimizer instance
        """

        # Optimizer parameter groups
        if layerwise_params is not None:
            if not apply_weight_decay_to_bias:
                raise ValueError("Layerwise params and no wd on bias are mutually exclusive")

            parameters = get_lr_decay_parameters(model, optimizer_params["lr"], layerwise_params)
        else:
            if apply_weight_decay_to_bias:
                parameters = get_optimizable_parameters(model)
            else:
                default_pg, biases_pg = [], []

                for k, v in model.named_parameters():
                    if v.requires_grad:
                        if str.endswith(k, ".bias"):
                            biases_pg.append(v)  # biases
                        else:
                            default_pg.append(v)  # all else

                if apply_weight_decay_to_bias:
                    parameters = default_pg + biases_pg
                else:
                    parameters = default_pg

        optimizer_cls = get_optimizer_cls(optimizer_name)
        optimizer: Optimizer = optimizer_cls(
            parameters,
            **optimizer_params,
        )

        if not apply_weight_decay_to_bias:
            optimizer.add_param_group({"params": biases_pg, "weight_decay": 0.0})

        return optimizer

    def build_model(self, config: Dict) -> nn.Module:
        train_config: Dict = self.cfg["train"]

        model = self.get_model(config["model"])

        if self.cfg.transfer:
            transfer_checkpoint = fs.auto_file(self.cfg.transfer)
            self.master_print("Transferring weights from model checkpoint", transfer_checkpoint)
            checkpoint = load_checkpoint(transfer_checkpoint)
            pretrained_dict = checkpoint["model_state_dict"]

            transfer_weights(model, pretrained_dict)
        elif self.cfg.checkpoint:
            checkpoint = load_checkpoint(fs.auto_file(self.cfg.checkpoint))
            unpack_checkpoint(checkpoint, model=model)

            self.master_print("Loaded model weights from:", self.cfg.checkpoint)
            report_checkpoint(checkpoint)

        freeze_encoder = train_config.get("freeze_encoder", False)
        if freeze_encoder:
            freeze_model(model.encoder, freeze_parameters=True, freeze_bn=False)
            self.master_print("Frozen model encoder")

        model = model.cuda()
        if self.cfg.torch.channels_last:
            model = model.to(memory_format=torch.channels_last)
            self.master_print("Using Channels Last")
        return model

    def get_experiment_dir(self, config) -> str:
        if self.cfg.experiment:
            return self.cfg.experiment

        current_time = datetime.now().strftime("%y%m%d_%H_%M")
        experiment_slug = f"{current_time}_{config.model.config.slug}_{config.dataset.slug}_{config.loss.slug}_{config.augs.slug}"

        fold = config["dataset"].get("fold", None)
        if fold is not None:
            experiment_slug += f"_fold{fold}"

        log_dir = os.path.join("runs", experiment_slug)
        return log_dir

    def get_model(self, config: DictConfig) -> nn.Module:
        from hydra.utils import instantiate

        return instantiate(config, _recursive_=False)

    def train(self):
        config = self.cfg
        model = self.build_model(config)
        loaders, dataset_callbacks = self.build_loaders(config)
        optimizer, scheduler, optimizer_callbacks = self.build_optimizer(model, loaders)
        criterions, criterions_callbacks = self.build_criterions(config)
        metric_callbacks = self.build_metrics(config, loaders, model)

        experiment_dir = self.get_experiment_dir(config)
        self.experiment_dir = experiment_dir

        if self.is_master:
            os.makedirs(experiment_dir, exist_ok=True)
            dst_config_fname = os.path.join(experiment_dir, "config.yaml")
            with open(dst_config_fname, "w") as f:
                OmegaConf.save(self.cfg, f)

        # model training
        runner_config = config.get("runner", {})
        input_key = runner_config.get("input_key", INPUT_IMAGE_KEY)
        output_key = runner_config.get("output_key", None)

        if input_key is not None and not isinstance(input_key, str):
            input_key = OmegaConf.to_container(input_key, resolve=True)
        if output_key is not None and not isinstance(output_key, str):
            output_key = OmegaConf.to_container(output_key, resolve=True)

        runner = SupervisedRunner(input_key=input_key, output_key=output_key, device="cuda")
        extra_callbacks = [TimerCallback()]
        # try:
        runner.train(
            fp16=self.build_distributed_params(config),
            model=model,
            criterion=criterions,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=dataset_callbacks + metric_callbacks + optimizer_callbacks + criterions_callbacks + extra_callbacks,
            loaders=loaders,
            logdir=experiment_dir,
            num_epochs=config.train.epochs,
            verbose=True,
            main_metric=config.runner.main_metric,
            minimize_metric=config.runner.main_metric_minimize,
            checkpoint_data=self.get_checkpoint_data(),
        )
        # except Exception as e:
        #     with open(os.path.join(experiment_dir, "exception.log"), "w") as logf:
        #         logf.write(str(e))
        #         traceback.print_exc(file=logf)
        #         print(e)

        self.on_experiment_finished()

    def get_checkpoint_data(self):
        return {
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "version": get_version(),
        }

    def on_experiment_finished(self):
        pass

    def build_quircks(self, config: DictConfig) -> List[Callback]:
        extra_callbacks = []

        if config.ema.callback is not None:
            cb = hydra.utils.instantiate(config.ema.callback)
            extra_callbacks.append(cb)
            self.master_print("Using EMA Callback", cb)

        return extra_callbacks


class TrainOnlyCriterionCallback(CriterionCallback):
    def on_batch_end(self, runner) -> None:
        if runner.is_train_loader:
            return super(TrainOnlyCriterionCallback, self).on_batch_end(runner)
        else:
            runner.batch_metrics[self.prefix] = torch.tensor(0, device="cuda")
