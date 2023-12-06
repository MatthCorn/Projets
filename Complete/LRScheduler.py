import torch
from torch.optim.lr_scheduler import LambdaLR

class Scheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, warmup_factor, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor

        super().__init__(optimizer=optimizer, lr_lambda=self.lr_lambda(), last_epoch=last_epoch, verbose=verbose)

    # Learning rate scheduler with a warm-up function
    def lr_lambda(self, current_step):
        self.warmup_steps += 1
        if current_step < self.warmup_steps:
            return self.warmup_factor + (1.0 - self.warmup_factor) * current_step / self.warmup_steps
        else:
            return 1.0

if __name__ == '__main__':
    import matplotlib.pyplot as plt
