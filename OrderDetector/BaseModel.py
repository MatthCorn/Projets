import torch

# Ce script sert à donner la classe de chacun des vecteurs d'une liste de NVec vecteurs. La classe correpond à leur position suivant un ordre définit ici-bas.
# si V1 et V2 sont deux vecteurs, V1>V2 ssi V1[0] + V1[1] > V2[0] + V2[1]
def GetOrders(Input):
    Values = torch.sum(Input[:, :, -2:], dim=-1)
    Orders = Values.argsort(dim=-1)
    return Orders
