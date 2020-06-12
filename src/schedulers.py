from typing import Any, Optional

import numpy as np
import torch


class ReduceOnPlateauWithWarmup(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(
            self, optimizer, warmup_steps: int, factor: float, patience: int,
            global_step: int, mode: str = 'min'
    ):
        super().__init__(optimizer=optimizer, factor=factor, patience=patience, mode=mode)
        self._warmup_steps = warmup_steps
        self._warmup_multipliers = iter(range(global_step + 1, self._warmup_steps + 1))
        self._base_lrs = [float(param_group['lr']) for param_group in getattr(optimizer, 'param_groups')]
        self.optimizer = optimizer
        self._metrics = np.inf
        self._step()

    def step(self, metrics: Any, epoch: Optional[int] = ...) -> None:
        self._step(metrics=metrics)

    def _step(self, metrics=None):
        try:
            warmup_multiplier = next(self._warmup_multipliers)
        except StopIteration:
            if metrics is not None and self._metrics != metrics:
                self._metrics = metrics
                super().step(metrics=self._metrics)
        else:
            for i, param_group in enumerate(getattr(self.optimizer, 'param_groups')):
                param_group['lr'] = self._base_lrs[i] * (warmup_multiplier / self._warmup_steps)
