import torch

def GetSelected(Input, WeightsCut, NOutput):
    if WeightsCut is None:
        WeightsCut = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightsCut.to(Input.device))
    Orders = Values.argsort(dim=-1)

    batch_size, NInput, Dim = Input.shape

    OrdersExpanded = Orders.unsqueeze(-1).expand(batch_size, NInput, Dim)

    Selected = Input[OrdersExpanded < NOutput].reshape(batch_size, NOutput, Dim)
    Others = Input[OrdersExpanded >= NOutput].reshape(batch_size, NInput-NOutput, Dim)
    return Selected, Others, Orders

def GetSorted(Input, WeightsSort):
    if WeightsSort is None:
        WeightsSort = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightsSort.to(Input.device))
    Orders = Values.argsort(dim=-1)
    Output = Input[torch.arange(Input.size(0)).unsqueeze(1), Orders]

    return Output, Orders


# Les coordonnées intervenants dans l'ordre sont centrées et d'écart type sigma. On applique ensuite une translation sur ces coordonnées.
# L'objectif est de voir la plage de fonctionnement de l'IA sur le tri, surtout voir la conséquence d'une forte translation

def MakeData(NInput=5, DVec=10, sigma=1, NData=1000, WeightsCut=None, WeightsSort=None, NOutput=5):
    Input = torch.normal(torch.zeros(NData, NInput, DVec), sigma*torch.ones(NData, NInput, DVec))

    ShiftInterval = [0, 10]
    Shift = ((ShiftInterval[0] + (ShiftInterval[1] - ShiftInterval[0])*torch.rand(size=(NData, 1, 1))).expand(NData, NInput, DVec))

    Input += Shift

    Selected, Others, FirstOrders = GetSelected(Input, WeightsCut, NOutput)
    Selected, SelectedOrders = GetSorted(Selected, WeightsSort)
    Others, OthersOrders = GetSorted(Others, WeightsSort)
    FinalOrders = torch.zeros(FirstOrders.shape, dtype=torch.int64)
    FinalOrders[FirstOrders < NOutput] = SelectedOrders.reshape(-1)
    FinalOrders[FirstOrders >= NOutput] = OthersOrders.reshape(-1) + NOutput
    return Input, FinalOrders.unsqueeze(-1).to(torch.float)


