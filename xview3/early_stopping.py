from collections import defaultdict
from typing import Union, Iterable

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder

__all__ = ["EarlyStoppingCallback"]


class EarlyStoppingCallback(Callback):
    """Early exit based on metric.

    Example of usage in notebook API:

    .. code-block:: python

        runner = SupervisedRunner()
        runner.train(
            ...
            callbacks=[
                ...
                EarlyStoppingCallback(
                    patience=5,
                    metric="my_metric",
                    minimize=True,
                )
                ...
            ]
        )
        ...

    Example of usage in config API:

    .. code-block:: yaml

        stages:
          ...
          stage_N:
            ...
            callbacks_params:
              ...
              early_stopping:
                callback: EarlyStoppingCallback
                # arguments for EarlyStoppingCallback
                patience: 5
                metric: my_metric
                minimize: true
          ...

    """

    def __init__(
        self,
        patience: int,
        metrics: Union[str, Iterable[str]] = "loss",
        minimize: bool = True,
        min_delta: float = 1e-6,
    ):
        """
        Args:
            patience: number of epochs with no improvement
                after which training will be stopped.
            metric: metric name to use for early stopping, default
                is ``"loss"``.
            minimize: if ``True`` then expected that metric should
                decrease and early stopping will be performed only when metric
                stops decreasing. If ``False`` then expected
                that metric should increase. Default value ``True``.
            min_delta: minimum change in the monitored metric
                to qualify as an improvement, i.e. an absolute change
                of less than min_delta, will count as no improvement,
                default value is ``1e-6``.
        """
        super().__init__(order=CallbackOrder.external, node=CallbackNode.all)
        self.best_score = defaultdict(lambda: None)
        self.metrics = [metrics] if isinstance(metrics, str) else list(metrics)
        self.patience = patience
        self.num_bad_epochs = 0
        self.is_better = None

        if minimize:
            self.is_better = lambda score, best: score <= (best - min_delta)
        else:
            self.is_better = lambda score, best: score >= (best + min_delta)

    def on_epoch_end(self, runner: "IRunner") -> None:
        """Check if should be performed early stopping.

        Args:
            runner: current runner
        """
        if runner.stage_name.startswith("infer"):
            return

        some_metric_has_improved = False

        for metric_name in self.metrics:
            score = runner.valid_metrics[metric_name]
            if self.best_score[metric_name] is None or self.is_better(score, self.best_score[metric_name]):
                self.best_score[metric_name] = float(score)
                some_metric_has_improved = True

        if some_metric_has_improved:
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print(f"Early stop at {runner.epoch} epoch")
            runner.need_early_stop = True
