import torch
from torch.nn import functional as F
import math

def GetOrders(Input):
    Values = torch.sum(Input[:, :, :2], dim=-1)
    Orders = Values.argsort(dim=-1)
    return Orders


# Les coordonnées intervenants dans l'ordre sont centrées et d'écart type sigma. On applique ensuite une translation sur ces coordonnées.
# L'objectif est de voir la plage de fonctionnement de l'IA sur le tri, surtout voir la conséquence d'une forte translation

NVec = 5
DVec = 10

sigma = 1
NData = 1000
TrainingInput = torch.normal(torch.zeros(NData, NVec, DVec), sigma*torch.ones(NData, NVec, DVec))

ShiftRange = 100
Shift = torch.exp(math.log(ShiftRange)*torch.rand(size=(NData, 1, 1)).expand(NData, NVec, 2))
Shift = F.pad(Shift, (0, DVec-2))

TrainingInput += Shift
TrainingOutPut = GetOrders(TrainingInput)

