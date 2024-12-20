import torch

def GetSelected(Input, WeightF, Threshold=0.14):
    if WeightF is None:
        WeightF = torch.tensor([1.]*Input.shape[-1])
    ValuesF = torch.matmul(Input, WeightF)
    DistF = torch.abs(ValuesF.unsqueeze(-1) - ValuesF.unsqueeze(-2)) < Threshold
    InterF = torch.sum(DistF, dim=-2)
    Mask = InterF > 1

    return Mask

def GetSorted(Input, Mask, WeightN):
    if WeightN is None:
        WeightN = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightN)
    Values[Mask] = -float('inf')
    Orders = Values.argsort(dim=-1, descending=True)
    Output = Input.clone()
    Output[Mask] = 0

    return Output[torch.arange(Output.size(0)).unsqueeze(1), Orders]


def MakeData(NInput=10, DVec=10, mean=0, sigma=1, NData=1000, WeightF=None, WeightN=None, Threshold=0.14):
    Input = torch.normal(mean, sigma, (NData, NInput, DVec))

    Mask = GetSelected(Input, WeightF, Threshold=Threshold)
    Output = GetSorted(Input, Mask, WeightN)
    return Input, Output