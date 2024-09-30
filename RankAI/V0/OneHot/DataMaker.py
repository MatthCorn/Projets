import torch
from torch.nn import functional as F

def GetRanks(Input, Weight):
    if Weight is None:
        Weight = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, Weight.to(Input.device))
    Orders = Values.argsort(dim=-1)
    return Orders

def MakeData(NVec=5, DVec=10, sigma=1, NData=1000, Weight=None, device=torch.device('cpu')):
    Input = torch.normal(0, sigma, (NData, NVec, DVec))

    Output = GetRanks(Input, Weight)
    Output = F.one_hot(Output, num_classes=NVec)
    return Input.to(device), Output.to(device, float)


