import torch

def GetSelected(Input, WeightCut, NOutput):
    Indices = torch.arange(0, Input.size(1)).unsqueeze(0).expand(Input.size(0), Input.size(1))
    if WeightCut is None:
        WeightCut = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightCut.to(Input.device))
    Orders = Values.argsort(dim=-1)
    Output = Input[torch.arange(Input.size(0)).unsqueeze(1), Orders]
    SelectedIndices = Indices[torch.arange(Input.size(0)).unsqueeze(1), Orders]
    return Output[:, :NOutput], SelectedIndices[:, :NOutput]

def GetSorted(SelectedVectors, SelectedIndices, WeightSort):
    if WeightSort is None:
        WeightSort = torch.tensor([1.]*SelectedVectors.shape[-1])
    Values = torch.matmul(SelectedVectors, WeightSort.to(SelectedVectors.device))
    Orders = Values.argsort(dim=-1)
    Output = SelectedIndices[torch.arange(SelectedIndices.size(0)).unsqueeze(1), Orders]

    return Output


def MakeData(NInput=5, DVec=10, mean=0, sigma=1, NData=1000, WeightCut=None, WeightSort=None, NOutput=5, device=torch.device('cpu')):
    Input = torch.normal(mean, sigma, (NData, NInput, DVec))

    SelectedVectors, SelectedIndices = GetSelected(Input, WeightCut, NOutput)
    Output = GetSorted(SelectedVectors, SelectedIndices, WeightSort).unsqueeze(-1)
    return Input.to(device), Output.to(device, float)


