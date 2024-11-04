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

    Model = torch.nn.Sequential(torch.nn.Linear(5, 10), Dropout(p=0.5), torch.nn.ReLU(),
                                torch.nn.Linear(10, 10), Dropout(p=0.5), torch.nn.ReLU(),
                                torch.nn.Linear(10, 10), Dropout(p=0.5), torch.nn.ReLU(),
                                torch.nn.Linear(10, 5))

    vec = torch.normal(0, 1, (5, 5))

    optim = torch.optim.Adam(Model.parameters(), lr=3e-4)
    batch_size = 5000
    dim = 5
    err_list = []
    err_list2 = []
    for i in tqdm(range(1000)):
        input = torch.normal(0, 1, (batch_size, dim))
        output = input @ vec

        N = 4
        prediction = Model(input.repeat(N, 1))

        prediction = prediction.reshape(N, batch_size, -1).permute(1, 2, 0)

        pred_mean = torch.mean(prediction, dim=-1)
        pred_red = prediction - pred_mean.unsqueeze(-1)
        pred_cov = (pred_red @ pred_red.transpose(1, 2)) / (N - 1)
        pred_inv_cov = torch.linalg.inv(pred_cov + 1e-5 * torch.eye(dim, dim).unsqueeze(0).expand(batch_size, -1 ,-1))

        err = torch.mean((output - pred_mean).unsqueeze(-1).transpose(1, 2) @ pred_inv_cov @ (output - pred_mean).unsqueeze(-1))

        pred = Model(input)
        err2 = torch.norm(pred-output)

        optim.zero_grad(set_to_none=True)
        err.backward()
        optim.step()


        err_list2.append(float(err2))
        err_list.append(float(err))

    plt.plot(err_list, 'r')
    plt.plot(err_list2, 'b')
    plt.show()
