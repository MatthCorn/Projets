from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class Scheduler(LambdaLR):
    def __init__(self,
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 max: float = 1.,
                 last_epoch: int = -1,
                 verbose: bool = False) -> None:
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.max = max

        super().__init__(optimizer, self.calc_lr, last_epoch, verbose)
    def calc_lr(self, step):
        if step == 0:
            return 0
        # return self.dim_embed ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        return self.max * min(step / self.warmup_steps, (step / self.warmup_steps) ** (-0.5))