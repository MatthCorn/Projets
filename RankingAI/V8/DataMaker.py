import torch
import torch.nn.functional as F

def GetSelected(Input, WeightsCut, LimCut=0):
    if WeightsCut is None:
        WeightsCut = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightsCut.to(Input.device))
    Mask = Values > LimCut

    return Mask

def GetSorted(Input, Mask, WeightsSort):
    if WeightsSort is None:
        WeightsSort = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightsSort.to(Input.device))
    Values[Mask] = float('inf')
    Orders = Values.argsort(dim=-1).argsort(dim=-1)
    Orders[Mask] = -1

    return F.one_hot(Orders + 1, num_classes=Orders.size(-1) + 1)


# Les coordonnées intervenants dans l'ordre sont centrées et d'écart type sigma. On applique ensuite une translation sur ces coordonnées.
# L'objectif est de voir la plage de fonctionnement de l'IA sur le tri, surtout voir la conséquence d'une forte translation

def MakeData(NInput=5, DVec=10, sigma=1, NData=1000, WeightsCut=None, WeightsSort=None, LimCut=0):
    Input = torch.normal(torch.zeros(NData, NInput, DVec), sigma*torch.ones(NData, NInput, DVec))

    Mask = GetSelected(Input, WeightsCut, LimCut)
    Class = GetSorted(Input, Mask, WeightsSort)
    return Input, Class.to(torch.float)


