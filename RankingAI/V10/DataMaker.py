import torch

def GetSelected(Input, WeightsCut, LimCut=0, LenOut=5):
    if WeightsCut is None:
        WeightsCut = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightsCut)
    Orders = Values.argsort(dim=-1).argsort(dim=-1)
    MaskOrders = Orders > LenOut
    Mask = Values > LimCut
    Mask[MaskOrders] = True

    return Mask

def GetSorted(Input, Mask, WeightsSort, LenOut):
    if WeightsSort is None:
        WeightsSort = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightsSort)
    Values[Mask] = float('inf')
    Orders = Values.argsort(dim=-1)
    Output = Input
    Output[Mask] = 0

    return Output[torch.arange(Output.size(0)).unsqueeze(1), Orders]


def MakeData(NInput=5, DVec=10, sigma=1, NData=1000, WeightsCut=None, WeightsSort=None, LenOut=5):
    Input = torch.normal(torch.zeros(NData, NInput, DVec), sigma*torch.ones(NData, NInput, DVec))

    Mask = GetSelected(Input, WeightsCut, LimCut=0, LenOut=LenOut)
    Output = GetSorted(Input, Mask, WeightsSort, LenOut=LenOut)
    return Input, Output[:, :LenOut]


