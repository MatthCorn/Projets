import torch

def GetSelected(Input, WeightCut, LimCut=0, NOutput=5):
    if WeightCut is None:
        WeightCut = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightCut)
    Orders = Values.argsort(dim=-1).argsort(dim=-1)
    MaskOrders = Orders > NOutput
    Mask = Values > LimCut
    Mask[MaskOrders] = True

    return Mask

def GetSorted(Input, Mask, WeightSort):
    if WeightSort is None:
        WeightSort = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightSort)
    Values[Mask] = float('inf')
    Orders = Values.argsort(dim=-1)
    Output = Input
    Output[Mask] = 0

    return Output[torch.arange(Output.size(0)).unsqueeze(1), Orders]


def MakeData(NInput=5, DVec=10, sigma=1, NData=1000, WeightCut=None, WeightSort=None, NOutput=5):
    Input = torch.normal(torch.zeros(NData, NInput, DVec), sigma*torch.ones(NData, NInput, DVec))

    Mask = GetSelected(Input, WeightCut, LimCut=0, NOutput=NOutput)
    Output = GetSorted(Input, Mask, WeightSort)
    return Input, Output[:, :NOutput]


