import torch
from torch.nn import functional as F
import math

def GetRanks(Input, Weight):
    if Weight is None:
        Weight = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, Weight.to(Input.device))
    Orders = Values.argsort(dim=-1)
    return Orders

def MakeData(NVec=5, DVec=10, mean=0, sigma=1, NData=1000, Weight=None, device=torch.device('cpu')):
    Input = torch.normal(mean, sigma, (NData, NVec, DVec))

    Output = GetRanks(Input, Weight)
    Output = F.one_hot(Output, num_classes=NVec)
    return Input.to(device), Output.to(device, float)

def MakeTargetedData(NVec=5, DVec=10, mean_min=1e-1, mean_max=1e1, sigma_min=1e0, sigma_max=1e1, distrib='log', plot=False, NData=1000, Weight=None, device=torch.device('cpu')):
    if Weight is None:
        Weight = torch.tensor([1.] * DVec)
    Weight = Weight / torch.norm(Weight)
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
    sigma = g((f(sigma_max) - f(sigma_min)) * spacing(NData) + f(sigma_min))

    if plot:
        mean, sigma = torch.meshgrid(mean, sigma)
        NData = NData ** 2

    mean, sigma = mean.reshape(-1, 1), sigma.reshape(-1, 1)

    alpha = torch.normal(0, 1, (NData, NVec)) * sigma + mean

    Input = torch.normal(0, 1, (NData, NVec, DVec)) * alpha.unsqueeze(-1)
    uncontroled_values = torch.matmul(Input, Weight.to(Input.device))
    Input = Input + (alpha - uncontroled_values).unsqueeze(-1) * Weight.view(1, 1, DVec)

    Output = GetRanks(Input, Weight)
    Output = F.one_hot(Output, num_classes=NVec)
    return Input.to(device), Output.to(device, float)
