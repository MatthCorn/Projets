import torch

def GetSelected(Input, WeightCut, NOutput):
    if WeightCut is None:
        WeightCut = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightCut.to(Input.device))
    Orders = Values.argsort(dim=-1)
    Output = Input[torch.arange(Input.size(0)).unsqueeze(1), Orders]
    return Output[:, :NOutput]

def GetSorted(Input, WeightSort):
    if WeightSort is None:
        WeightSort = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightSort.to(Input.device))
    Orders = Values.argsort(dim=-1)
    Output = Input[torch.arange(Input.size(0)).unsqueeze(1), Orders]

    return Output


def MakeData(NInput=5, DVec=10, mean=0, sigma=1, NData=1000, WeightCut=None, WeightSort=None, NOutput=5):
    Input = torch.normal(mean, sigma, (NData, NInput, DVec))

    Selected = GetSelected(Input, WeightCut, NOutput)
    Output = GetSorted(Selected, WeightSort)
    return Input, Output
