import torch

def GetSelected(Input, WeightsF, WeightsN, Threshold=0.14):
    if WeightsF is None:
        WeightsF = torch.tensor([1.]*Input.shape[-1])
    if WeightsN is None:
        WeightsN = torch.tensor([1.] * Input.shape[-1])
    ValuesF = torch.matmul(Input, WeightsF)
    DistF = torch.abs(ValuesF.unsqueeze(-1) - ValuesF.unsqueeze(-2)) < Threshold
    ValuesN = torch.matmul(Input, WeightsN)
    SupN = (ValuesN.unsqueeze(-1) > ValuesN.unsqueeze(-2))
    InterF = torch.sum(DistF * SupN, dim=-2)
    Mask = InterF > 0

    return Mask

def GetSorted(Input, Mask, WeightsSort):
    if WeightsSort is None:
        WeightsSort = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightsSort)
    Values[Mask] = -float('inf')
    Orders = Values.argsort(dim=-1, descending=True)
    Output = Input.clone()
    Output[Mask] = 0

    return Output[torch.arange(Output.size(0)).unsqueeze(1), Orders]


def MakeData(NInput=10, DVec=10, sigma=1, NData=1000, WeightsF=None, WeightsN=None, Threshold=0.14):
    Input = torch.normal(torch.zeros(NData, NInput, DVec), sigma*torch.ones(NData, NInput, DVec))

    Mask = GetSelected(Input, WeightsF, WeightsN, Threshold=Threshold)
    Output = GetSorted(Input, Mask, WeightsN)
    return Input, Output


if __name__ == '__main__':
    MakeData(10, 2, 1, 1, WeightsF=torch.tensor([1., 0.]), WeightsN=torch.tensor([0., 1.]), Threshold=0.14)

