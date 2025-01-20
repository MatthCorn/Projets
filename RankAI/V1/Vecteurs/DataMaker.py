import torch
import math

def GetSelected(Input, WeightCut, NOutput):
    if WeightCut is None:
        WeightCut = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightCut.to(Input.device))
    Orders = Values.argsort(dim=-1)
    Output = Input[torch.arange(Input.size(0)).unsqueeze(1), Orders]
    return Output[:, :NOutput]

def GetSorted(Input, WeightSort):
    if WeightSort is None:
        WeightSort = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightSort.to(Input.device))
    Orders = Values.argsort(dim=-1)
    Output = Input[torch.arange(Input.size(0)).unsqueeze(1), Orders]

    return Output


def MakeData(NInput=5, DVec=10, mean=0, std=1, NData=1000, WeightCut=None, WeightSort=None, NOutput=5):
    Input = torch.normal(mean, std, (NData, NInput, DVec))

    Selected = GetSelected(Input, WeightCut, NOutput)
    Output = GetSorted(Selected, WeightSort)
    return Input, Output

def MakeTargetedData(NInput=10, NOutput=5, DVec=10, mean_min=1e-1, mean_max=1e1, std_min=1e0, std_max=1e1,
                     distrib='log', plot=False, NData=1000, WeightCut=None, WeightSort=None, device=torch.device('cpu')):
    if WeightCut is None:
        WeightCut = torch.tensor([1.] * DVec)
    if WeightSort is None:
        WeightSort = torch.tensor([1.] * DVec)
    WeightSort = WeightSort / torch.norm(WeightSort)
    WeightCut = WeightCut / torch.norm(WeightCut)
    scal = torch.dot(WeightSort, WeightCut)
    if plot:
        spacing = lambda x: torch.linspace(0, 1, x)
    else:
        spacing = lambda x: torch.rand(x)

    if distrib == 'uniform':
        f = lambda x: x
        g = lambda x: x
    elif distrib == 'log':
        f = lambda x: math.log(x)
        g = lambda x: torch.exp(x)
    elif distrib == 'exp':
        f = lambda x: math.exp(x)
        g = lambda x: torch.log(x)
    mean = g((f(mean_max) - f(mean_min)) * spacing(NData) + f(mean_min))
    std = g((f(std_max) - f(std_min)) * spacing(NData) + f(std_min))

    if plot:
        mean, std = torch.meshgrid(mean, std)
        NData = NData ** 2

    mean, std = mean.reshape(-1, 1), std.reshape(-1, 1)
    gamma_mean = torch.rand(*mean.size())
    eps_mean = torch.randint(0, 2, tuple(mean.shape))
    gamma_std = torch.rand(tuple(std.shape)) * (1 - abs(scal)) / (1 + abs(scal)) + abs(scal) / (1 + abs(scal))
    mean_alpha = (-1) ** (eps_mean + (mean > 0)) * abs(mean) * gamma_mean
    mean_beta = (-1) ** eps_mean * abs(mean) * (1 - gamma_mean)
    std_alpha = std * gamma_std
    std_beta = std * (1 - gamma_std)

    def ortho(x):
        v1 = WeightSort
        v2 = WeightCut - scal * WeightSort
        v2 = v2 / torch.norm(v2)
        p1 = x - torch.matmul(x, v1.to(x.device)).unsqueeze(-1) * v1.view(1, 1, DVec)
        p2 = p1 - torch.matmul(x, v2.to(x.device)).unsqueeze(-1) * v2.view(1, 1, DVec)
        return p2

    alpha = (torch.normal(0, 1, (NData, NInput)) *
             torch.sqrt((std_alpha ** 2 - scal ** 2 * std_beta ** 2) / (1 - scal ** 4)) +
             (mean_alpha - scal * mean_beta) / (1 - scal ** 2))
    beta = (torch.normal(0, 1, (NData, NInput)) *
            torch.sqrt((std_beta - scal ** 2 * std_alpha) / (1 - scal ** 4)) +
            (mean_beta - scal * mean_alpha) / (1 - scal ** 2))

    Input = (torch.normal(0, 1, (NData, NInput, DVec)) * alpha.unsqueeze(-1) + torch.normal(0, 1, (NData, NInput, DVec)) * beta.unsqueeze(-1)) / 2
    Input = ortho(Input) + alpha.unsqueeze(-1) * WeightSort.view(1, 1, DVec) + beta.unsqueeze(-1) * WeightCut.view(1, 1, DVec)

    Selected = GetSelected(Input, WeightCut, NOutput)
    Output = GetSorted(Selected, WeightSort)
    return Input.to(device), Output.to(device)