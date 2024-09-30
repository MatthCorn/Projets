import torch.nn.functional as F
import torch

def GetSelected(Input, WeightF, WeightN, Threshold=0.14):
    if WeightF is None:
        WeightF = torch.tensor([1.]*Input.shape[-1])
    if WeightN is None:
        WeightN = torch.tensor([1.] * Input.shape[-1])
    ValuesF = torch.matmul(Input, WeightF)
    DistF = torch.abs(ValuesF.unsqueeze(-1) - ValuesF.unsqueeze(-2)) < Threshold
    ValuesN = torch.matmul(Input, WeightN)
    SupN = (ValuesN.unsqueeze(-1) > ValuesN.unsqueeze(-2))
    InterF = torch.sum(DistF * SupN, dim=-2)
    Mask = InterF > 0

    return Mask

def GetSorted(Input, Mask, WeightN):
    if WeightN is None:
        WeightN = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightN)
    Values[Mask] = -float('inf')
    Orders = Values.argsort(dim=-1)
    Indices = torch.arange(0, Input.size(1)).unsqueeze(0).expand(Input.size(0), Input.size(1)).to(float)
    Indices[Mask] = -1
    return Indices[torch.arange(Input.size(0)).unsqueeze(1), Orders].unsqueeze(-1)

def MakeData(NInput=10, DVec=10, sigma=1, NData=1000, WeightF=None, WeightN=None, Threshold=0.14, NOutput=5):
    Input = torch.normal(torch.zeros(NData, NInput, DVec), sigma*torch.ones(NData, NInput, DVec))

    Mask = GetSelected(Input, WeightF, WeightN, Threshold=Threshold)
    Indices = GetSorted(Input, Mask, WeightN)
    return Input, Indices[:, :NOutput]+1

if __name__ == '__main__':
    MakeData(NInput=10, NOutput=5)