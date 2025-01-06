import torch
import math


def GetRanks(Input, Weight):
    if Weight is None:
        Weight = torch.tensor([1.] * Input.shape[-1])
    Values = torch.matmul(Input, Weight.to(Input.device))
    Orders = Values.argsort(dim=-1)
    return Orders


def MakeData(NVec=5, DVec=10, mean=0, sigma=1, NData=1000, Weight=None, device=torch.device('cpu')):
    Input = torch.normal(mean, sigma, (NData, NVec, DVec))

    Output = GetRanks(Input, Weight).unsqueeze(dim=-1)
    return Input.to(device), Output.to(device)


def MakeTargetedData(NVec=5, DVec=10, mean_min=0, mean_max=0, sigma_min=1, sigma_max=1, distrib='log', NData=1000, Weight=None, device=torch.device('cpu')):
    if distrib == 'unif':
        mean = (mean_max - mean_min) * torch.rand(NData, 1) + mean_min
        sigma = (sigma_max - sigma_min) * torch.rand(NData, 1) + sigma_min
    elif distrib == 'log':
        mean = torch.exp((math.log(mean_max) - math.log(mean_min)) * torch.rand(NData, 1) + math.log(mean_min))
        sigma = torch.exp((math.log(sigma_max) - math.log(sigma_min)) * torch.rand(NData, 1) + math.log(sigma_min))
    Input = torch.normal(mean.expand(*mean.size()[:-1], NVec), sigma.expand(*sigma.size()[:-1], NVec))

    Output = GetRanks(Input, Weight).unsqueeze(dim=-1)
    return Input.to(device), Output.to(device)
