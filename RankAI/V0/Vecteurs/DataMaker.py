import torch

def GetSorted(Input, Weight):
    if Weight is None:
        Weight = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, Weight.to(Input.device))
    Orders = Values.argsort(dim=-1)
    return Input[torch.arange(Input.size(0)).unsqueeze(1), Orders]

def MakeData(NVec=5, DVec=10, sigma=1, NData=1000, Weight=None):
    Input = torch.normal(0, sigma, (NData, NVec, DVec))

    Output = GetSorted(Input, Weight)
    return Input, Output


