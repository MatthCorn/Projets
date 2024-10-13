from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class Scheduler(LambdaLR):
    def __init__(self,
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 dropping_step_list: int = [-1],
                 dropping_factor: float = 10,
                 max: float = 1.,
                 last_epoch: int = -1,
                 verbose: bool = False) -> None:
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.dropping_step_list = dropping_step_list
        self.dropping_factor = dropping_factor
        self.max = max
        self.last_lr = 0
        self.target_lr = 0
        self.mult_fact = 1

        super().__init__(optimizer, self.calc_lr, last_epoch, verbose)

    def calc_lr(self, step):
        if step == 0:
            return 1e-5
        self.target_lr = self.max * min(step / self.warmup_steps, (step / self.warmup_steps) ** (-0.5))
        if step in self.dropping_step_list:
            self.mult_fact /= self.dropping_factor
        self.target_lr = self.mult_fact * self.target_lr
        new_lr = 0.9 * self.last_lr + 0.1 * self.target_lr
        self.last_lr = new_lr
        return new_lr



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch

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
    sch = Scheduler(opt, 1, 10, max=5, dropping_step_list=[100, 150])
    lr_list = []
    for i in range(500):
        opt.zero_grad()
        o = model(torch.normal(torch.zeros(30, 1)))
        o.norm().backward()
        opt.step()
        lr_list.append(sch.last_lr)
        sch.step()

        if i % 100 == 0:
            print(i)
            print(opt.state)


    plt.plot(lr_list)
    plt.show()
    None