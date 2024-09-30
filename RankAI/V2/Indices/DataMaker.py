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
    Indices = torch.arange(0, Input.size(1)).unsqueeze(0).expand(Input.size(0), Input.size(1)).to(float)
    Indices[Mask] = -1
    return Indices[torch.arange(Input.size(0)).unsqueeze(1), Orders].unsqueeze(-1)


def MakeData(NInput=5, DVec=10, sigma=1, NData=1000, WeightCut=None, WeightSort=None, NOutput=5):
    Input = torch.normal(torch.zeros(NData, NInput, DVec), sigma*torch.ones(NData, NInput, DVec))

    Mask = GetSelected(Input, WeightCut, LimCut=0, NOutput=NOutput)
    Indices = GetSorted(Input, Mask, WeightSort)
    return Input, Indices[:, :NOutput]


if __name__ == '__main__':
    MakeData(NInput=10, NOutput=5)