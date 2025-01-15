import torch
import torch.nn.functional as F
import math

def GetSelected(Input, WeightCut, NOutput=5):
    if WeightCut is None:
        WeightCut = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightCut)
    Orders = Values.argsort(dim=-1).argsort(dim=-1)
    MaskOrders = Orders > NOutput
    Mask = Values > Values.mean(dim=-1, keepdim=True)
    Mask[MaskOrders] = True

    return Mask

def GetSorted(Input, Mask, WeightSort):
    if WeightSort is None:
        WeightSort = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightSort)
    Values[Mask] = float('inf')
    Orders = Values.argsort(dim=-1)
    Indices = torch.arange(0, Input.size(1)).unsqueeze(0).expand(Input.size(0), Input.size(1)).to(float)
    Indices[Mask] = -1
    return Indices[torch.arange(Input.size(0)).unsqueeze(1), Orders].to(int)


def MakeData(NInput=5, DVec=10, mean=0, sigma=1, NData=1000, WeightCut=None, WeightSort=None, NOutput=5):
    Input = torch.normal(mean, sigma, (NData, NInput, DVec))

    Mask = GetSelected(Input, WeightCut, NOutput=NOutput)
    Indices = GetSorted(Input, Mask, WeightSort)
    return Input, F.one_hot(Indices[:, :NOutput]+1, NInput+1).to(float)

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

    def ortho(x):
        v1 = WeightSort
        v2 = WeightCut - torch.dot(WeightSort, WeightCut) * WeightSort
        v2 = v2 / torch.norm(v2)
        p1 = x - torch.matmul(x, v1.to(x.device)).unsqueeze(-1) * v1.view(1, 1, DVec)
        p2 = p1 - torch.matmul(x, v2.to(x.device)).unsqueeze(-1) * v2.view(1, 1, DVec)
        return p2

    alpha = torch.normal(0, 1, (NData, NInput)) * sigma / (1 + torch.dot(WeightSort, WeightCut) ** 2) + mean / (1 + torch.dot(WeightSort, WeightCut))
    beta = torch.normal(0, 1, (NData, NInput)) * sigma / (1 + torch.dot(WeightSort, WeightCut) ** 2) + mean / (1 + torch.dot(WeightSort, WeightCut))

    Input = torch.normal(0, 1, (NData, NInput, DVec)) * (alpha.unsqueeze(-1) + beta.unsqueeze(-1)) / 2
    Input = ortho(Input) + alpha.unsqueeze(-1) * WeightSort.view(1, 1, DVec) + beta.unsqueeze(-1) * WeightCut.view(1, 1, DVec)

    Mask = GetSelected(Input, WeightCut, LimCut=0, NOutput=NOutput)
    Indices = GetSorted(Input, Mask, WeightSort)
    return Input, F.one_hot(Indices[:, :NOutput]+1, NInput+1).to(float)

if __name__ == '__main__':
    MakeData(NInput=10, NOutput=5)