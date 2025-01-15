from torch.nn import functional as F
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
    OutputIndices = SelectedIndices[torch.arange(SelectedIndices.size(0)).unsqueeze(1), Orders]
    return OutputIndices


def MakeData(NInput=5, DVec=10, mean=0, std=1, NData=1000, WeightCut=None, WeightSort=None, NOutput=5, device=torch.device('cpu')):
    Input = torch.normal(mean, std, (NData, NInput, DVec))

    SelectedVectors, SelectedIndices = GetSelected(Input, WeightCut, NOutput)
    Output = GetSorted(SelectedVectors, SelectedIndices, WeightSort)
    return Input.to(device), F.one_hot(Output, num_classes=NInput).to(device, float)

def MakeTargetedData(NInput=10, NOutput=5, DVec=10, mean_min=1e-1, mean_max=1e1, std_min=1e0, std_max=1e1,
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
    std = g((f(std_max) - f(std_min)) * spacing(NData) + f(std_min))

    if plot:
        mean, std = torch.meshgrid(mean, std)
        NData = NData ** 2

    mean, std = mean.reshape(-1, 1), std.reshape(-1, 1)

    def ortho(x):
        v1 = WeightSort
        v2 = WeightCut - torch.dot(WeightSort, WeightCut) * WeightSort
        v2 = v2 / torch.norm(v2)
        p1 = x - torch.matmul(x, v1.to(x.device)).unsqueeze(-1) * v1.view(1, 1, DVec)
        p2 = p1 - torch.matmul(x, v2.to(x.device)).unsqueeze(-1) * v2.view(1, 1, DVec)
        return p2

    alpha = torch.normal(0, 1, (NData, NInput)) * std / (1 + torch.dot(WeightSort, WeightCut) ** 2) + mean / (1 + torch.dot(WeightSort, WeightCut))
    beta = torch.normal(0, 1, (NData, NInput)) * std / (1 + torch.dot(WeightSort, WeightCut) ** 2) + mean / (1 + torch.dot(WeightSort, WeightCut))

    Input = torch.normal(0, 1, (NData, NInput, DVec)) * (alpha.unsqueeze(-1) + beta.unsqueeze(-1)) / 2
    Input = ortho(Input) + alpha.unsqueeze(-1) * WeightSort.view(1, 1, DVec) + beta.unsqueeze(-1) * WeightCut.view(1, 1, DVec)

    SelectedVectors, SelectedIndices = GetSelected(Input, WeightCut, NOutput)
    Output = GetSorted(SelectedVectors, SelectedIndices, WeightSort)
    return Input.to(device), F.one_hot(Output, num_classes=NInput).to(device, float)
