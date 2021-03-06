import functools
from typing import Optional, Callable, Any

from hydra import TaskFunction
from hydra._internal.utils import get_args_parser, _run_hydra
from omegaconf import DictConfig

__all__ = ["hydra_dpp_friendly_main", "nan_value", "ignore_value"]

from xview3.constants import IGNORE_LABEL


def nan_value():
    return float("nan")


def ignore_value():
    return IGNORE_LABEL


def hydra_dpp_friendly_main(
    config_path: Optional[str] = None,
    config_name: Optional[str] = None,
    strict: Optional[bool] = None,
) -> Callable[[TaskFunction], Any]:
    """
    :param config_path: the config path, a directory relative to the declaring python file.
    :param config_name: the name of the config (usually the file name without the .yaml extension)
    :param strict: (Deprecated) strict mode, will throw an error if command line overrides are not changing an
    existing key or if the code is accessing a non existent key
    """

    def main_decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def decorated_main(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)
            else:
                args = get_args_parser()
                # Add local_rank to be able to use hydra with DDP
                args.add_argument("--local_rank", default=0, type=int)
                # no return value from run_hydra() as it may sometime actually run the task_function
                # multiple times (--multirun)
                _run_hydra(
                    args_parser=args,
                    task_function=task_function,
                    config_path=config_path,
                    config_name=config_name,
                )

        return decorated_main

    return main_decorator
