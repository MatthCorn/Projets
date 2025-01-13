import torch
import math

def GetSelected(Input, WeightCut, NOutput):
    Indices = torch.arange(0, Input.size(1)).unsqueeze(0).expand(Input.size(0), Input.size(1))
    if WeightCut is None:
        WeightCut = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightCut.to(Input.device))
    Orders = Values.argsort(dim=-1)
    Output = Input[torch.arange(Input.size(0)).unsqueeze(1), Orders]
    SelectedIndices = Indices[torch.arange(Input.size(0)).unsqueeze(1), Orders]
    return Output[:, :NOutput], SelectedIndices[:, :NOutput]

def GetSorted(SelectedVectors, SelectedIndices, WeightSort):
    if WeightSort is None:
        WeightSort = torch.tensor([1.]*SelectedVectors.shape[-1])
    Values = torch.matmul(SelectedVectors, WeightSort.to(SelectedVectors.device))
    Orders = Values.argsort(dim=-1)
    Output = SelectedIndices[torch.arange(SelectedIndices.size(0)).unsqueeze(1), Orders]

    return Output


def MakeData(NInput=5, DVec=10, mean=0, sigma=1, NData=1000, WeightCut=None, WeightSort=None, NOutput=5, device=torch.device('cpu')):
    Input = torch.normal(mean, sigma, (NData, NInput, DVec))

    SelectedVectors, SelectedIndices = GetSelected(Input, WeightCut, NOutput)
    Output = GetSorted(SelectedVectors, SelectedIndices, WeightSort).unsqueeze(-1)
    return Input.to(device), Output.to(device, float)

def MakeTargetedData(NInput=10, NOutput=5, DVec=10, mean_min=1e-1, mean_max=1e1, sigma_min=1e0, sigma_max=1e1,
                     distrib='log', plot=False, NData=1000, WeightCut=None, WeightSort=None, device=torch.device('cpu')):
    if WeightCut is None:
        WeightCut = torch.tensor([1.] * DVec)
    if WeightSort is None:
        WeightSort = torch.tensor([1.] * DVec)
    WeightSort = WeightSort / torch.norm(WeightSort)
    WeightCut = WeightCut / torch.norm(WeightCut)
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

    Output = GetRanks(Input, Weight).unsqueeze(dim=-1)
    return Input.to(device), Output.to(device)
