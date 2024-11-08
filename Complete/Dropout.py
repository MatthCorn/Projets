import torch
import torch.nn as nn

class Dropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = torch.bernoulli(torch.full_like(x, 1 - self.p)).to(x.device)
        return x * mask / (1 - self.p)

if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from Complete.LRScheduler import Scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Model = torch.nn.Sequential(torch.nn.Linear(5, 50), Dropout(p=0.1), torch.nn.ReLU(),
                                torch.nn.Linear(50, 50), Dropout(p=0.1), torch.nn.ReLU(),
                                torch.nn.Linear(50, 50), Dropout(p=0.1), torch.nn.ReLU(),
                                torch.nn.Linear(50, 5))
    Model.to(device)

    vec = torch.normal(0, 1, (5, 5), device=device)

    optim = torch.optim.Adam(Model.parameters(), lr=3e-5)
    sch = Scheduler(optim, 1, warmup_steps=10, max=1, max_steps=50000, ramp_steps=5000, dropping_step_list=[100, 150], type='cos')
    batch_size = 5000
    dim = 5
    err_list = []
    err_list2 = []
    for i in tqdm(range(50000)):
        input = torch.normal(0, 1, (batch_size, dim), device=device)
        output = input @ vec

        N = 4
        prediction = Model(input.repeat(N, 1))

        prediction = prediction.reshape(N, batch_size, -1).permute(1, 2, 0)

        pred_mean = torch.mean(prediction, dim=-1)
        pred_var = torch.var(prediction, dim=-1)
        pred_red = (prediction - pred_mean.unsqueeze(-1))#.detach().requires_grad_(False)
        pred_cov = (pred_red @ pred_red.transpose(1, 2)) / (N - 1)
        pred_cov_mod = pred_cov + 0.01 * torch.mean(abs(pred_cov)) * torch.eye(dim, dim, device=device)
        pred_cov_det_log = torch.log(torch.linalg.det(pred_cov_mod))
        pred_inv_cov = torch.linalg.inv(pred_cov_mod)

        err = torch.mean(0.5 * (output - pred_mean).unsqueeze(-1).transpose(1, 2) @ pred_inv_cov @ (output - pred_mean).unsqueeze(-1) + 0.5 * pred_cov_det_log)
        err = torch.mean(0.5 * (output - pred_mean).unsqueeze(-1).transpose(1, 2) @ (output - pred_mean).unsqueeze(-1) / torch.mean(pred_var, dim=-1)
                         + 0.5 * torch.prod(pred_var, dim=-1))

        pred = Model(input)
        err2 = torch.nn.functional.mse_loss(pred, output, reduction='mean')

        optim.zero_grad(set_to_none=True)
        (1e-2 * err).backward()
        optim.step()
        sch.step()


        err_list2.append(float(err2))
        err_list.append(float(err))

    plt.plot(err_list, 'r')
    plt.plot(err_list2, 'b')
    plt.ylim(0, 400)
    plt.show()
