from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right

# Based on https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py

class LinearMultiStepWarmUp(_LRScheduler):
    def __init__(self, cfg, optimizer, last_epoch=-1):
        self.gamma = cfg.SOLVER.GAMMA
        self.milestones = cfg.SOLVER.MULTISTEP_MILESTONES
        self.warmup_period = cfg.SOLVER.WARMUP_PERIOD

        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.warmup_period > 0.0:
            warmup_factor = min(1.0, (self._step_count+1)/self.warmup_period)
        else:
            warmup_factor = 1.0

        return [base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]
