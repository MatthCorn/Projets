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
    Orders = Values.argsort(dim=-1, descending=True)
    Output = Input.clone()
    Output[Mask] = 0

    return Output[torch.arange(Output.size(0)).unsqueeze(1), Orders]


def MakeData(NInput=10, DVec=10, mean=0, std=1, NData=1000, WeightF=None, WeightN=None, Threshold=0.14, NOutput=5):
    Input = torch.normal(mean, std, (NData, NInput, DVec))

    Mask = GetSelected(Input, WeightF, WeightN, Threshold=Threshold)
    Output = GetSorted(Input, Mask, WeightN)
    return Input, Output[:, :NOutput]