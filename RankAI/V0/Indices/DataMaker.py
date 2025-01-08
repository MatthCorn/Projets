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


def MakeTargetedData(NVec=5, DVec=10, mean_min=1e-1, mean_max=1e1, sigma_min=1e0, sigma_max=1e1, distrib='log', plot=False, NData=1000, Weight=None, device=torch.device('cpu')):
    if Weight is None:
        Weight = torch.tensor([1.] * DVec)
    Weight = Weight / torch.norm(Weight)
    if plot:
        if distrib == 'uniform':
            f = lambda x: x
        elif distrib == 'log':
            f = lambda x: math.log(x)
        mean = torch.exp((f(mean_max) - f(mean_min)) * torch.linspace(0, 1, NData) + f(mean_min))
        sigma = torch.exp((f(sigma_max) - f(sigma_min)) * torch.linspace(0, 1, NData) + f(sigma_min))

        mean, sigma = torch.meshgrid(mean, sigma)
        alpha = torch.normal(0, 1, (NData * NData, NVec)) * sigma.reshape(-1, 1) + mean.reshape(-1, 1)

        Input = torch.normal(0, 1, (NData * NData, NVec, DVec)) * sigma.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)
        uncontroled_values = torch.matmul(Input, Weight.to(Input.device))
        Input = Input + (alpha - uncontroled_values).unsqueeze(-1) * Weight.view(1, 1, DVec)

        Output = GetRanks(Input, Weight).unsqueeze(dim=-1)
        return Input.to(device), Output.to(device)
    else:
        if distrib == 'uniform':
            mean = (mean_max - mean_min) * torch.rand(NData, 1) + mean_min
            sigma = (sigma_max - sigma_min) * torch.rand(NData, 1) + sigma_min
        elif distrib == 'log':
            mean = torch.exp((math.log(mean_max) - math.log(mean_min)) * torch.rand(NData, 1) + math.log(mean_min))
            sigma = torch.exp((math.log(sigma_max) - math.log(sigma_min)) * torch.rand(NData, 1) + math.log(sigma_min))
        alpha = torch.normal(0, 1, (NData, NVec)) * sigma + mean

        Input = torch.normal(0, 1, (NData, NVec, DVec)) * sigma.unsqueeze(-1) + mean.unsqueeze(-1)
        uncontroled_values = torch.matmul(Input, Weight.to(Input.device))
        Input = Input + (alpha - uncontroled_values).unsqueeze(-1) * Weight.view(1, 1, DVec)

        Output = GetRanks(Input, Weight).unsqueeze(dim=-1)
        return Input.to(device), Output.to(device)


if __name__ == '__main__':
    MakeTargetedData(plot=True)