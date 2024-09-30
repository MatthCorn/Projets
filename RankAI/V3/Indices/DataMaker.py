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
    Indices = torch.arange(0, Input.size(1)).unsqueeze(0).expand(Input.size(0), Input.size(1)).to(float)
    Indices[Mask] = -1

    return Indices[torch.arange(Input.size(0)).unsqueeze(1), Orders]


def MakeData(NInput=10, DVec=10, sigma=1, NData=1000, WeightF=None, WeightN=None, Threshold=0.14):
    Input = torch.normal(torch.zeros(NData, NInput, DVec), sigma*torch.ones(NData, NInput, DVec))

    Mask = GetSelected(Input, WeightF, Threshold=Threshold)
    Indices = GetSorted(Input, Mask, WeightN)
    return Input, Indices.unsqueeze(-1)

if __name__ == '__main__':
    MakeData()