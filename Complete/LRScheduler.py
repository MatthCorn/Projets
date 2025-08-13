from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math


class Scheduler(LambdaLR):
    def __init__(self,
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 max_steps: int = 1000,
                 ramp_steps: int = 100,
                 max: float = 1.,
                 type=None,
                 last_epoch: int = -1,
                 last_lr = 0,
                 target_lr = 0) -> None:
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.ramp_steps = ramp_steps
        self.type = type
        self.max = max
        self.last_lr = last_lr
        self.target_lr = target_lr

        super().__init__(optimizer, self.calc_lr, last_epoch)

    def calc_lr(self, step):
        step += 1

        if self.type == 'cos':
            self.last_lr = self.max * min(step / self.warmup_steps,
                                  math.cos(math.pi / 2 * ((step - self.warmup_steps) / (1e-10 + self.max_steps - self.warmup_steps))) ** 2)
            return self.last_lr

        if self.type == 'ramp':
            if step - 1 < self.warmup_steps:
                self.last_lr = self.max * step / self.warmup_steps
            elif step - 1 > self.max_steps - self.ramp_steps:
                self.last_lr = self.max * (1 - (step - (self.max_steps - self.ramp_steps)) / self.ramp_steps)
            else:
                self.last_lr = self.max
            return self.last_lr

        self.target_lr = self.max * min(step / self.warmup_steps, (step / self.warmup_steps) ** (-0.5))
        new_lr = 0.95 * self.last_lr + 0.05 * self.target_lr
        self.last_lr = new_lr
        return new_lr

    def get_hparams(self):
        return {
            "dim_embed": self.dim_embed,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "ramp_steps": self.ramp_steps,
            "max": self.max,
            "type": self.type,
            "last_lr": self.last_lr,
            "target_lr": self.target_lr,
            "last_epoch": self.last_epoch
        }


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    from tqdm import tqdm

    class model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(1, 4, bias=False)
            self.fc2 = torch.nn.Linear(4, 3, bias=False)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    class modeltest(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model()
            self.fc1 = torch.nn.Linear(3, 3, bias=False)

        def forward(self, x):
            return self.fc1(self.model(x))

    # model = torch.nn.Sequential(torch.nn.Linear(1, 4, bias=False), torch.nn.Linear(4, 3, bias=False))
    model = modeltest()
    opt = torch.optim.Adam(model.parameters())
    sch = Scheduler(opt, 256, warmup_steps = int(500000 / 1000 * 2), max=5)
    n_updates = int(500000 / 1000) * 200
    # sch = Scheduler(opt, 1, warmup_steps=10, max=5, max_steps=100, ramp_steps=15, dropping_step_list=[100, 150], type='ramp')
    lr_list = []
    for i in tqdm(range(n_updates)):
        opt.zero_grad()
        o = model(torch.normal(torch.zeros(30, 1)))
        o.norm().backward()
        opt.step()
        lr_list.append(sch.last_lr)
        sch.step()

        # if i % 100 == 0:
        #     print(i)
        #     print(opt.state)


    plt.plot(lr_list)
    plt.show()
    None