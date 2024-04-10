import torch

def GetRanks(Input, weights):
    if weights is None:
        weights = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, weights.to(Input.device))
    Orders = Values.argsort(dim=-1)
    return Orders.unsqueeze(dim=-1)

def GetSorted(Input, weights):
    if weights is None:
        weights = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, weights.to(Input.device))
    Orders = Values.argsort(dim=-1)
    return Input[torch.arange(Input.size(0)).unsqueeze(1), Orders]


# Les coordonnées intervenants dans l'ordre sont centrées et d'écart type sigma. On applique ensuite une translation sur ces coordonnées.
# L'objectif est de voir la plage de fonctionnement de l'IA sur le tri, surtout voir la conséquence d'une forte translation

def MakeData(NVec=5, DVec=10, sigma=1, NData=1000, ShiftInterval=[0, 100], type='Ranks', weights=None, device=torch.device('cpu')):
    Input = torch.normal(torch.zeros(NData, NVec, DVec), sigma*torch.ones(NData, NVec, DVec))

    # Shift = torch.exp(math.log(ShiftRange)*torch.rand(size=(NData, 1, 1)).expand(NData, NVec, 2))
    Shift = ((ShiftInterval[0] + (ShiftInterval[1] - ShiftInterval[0])*torch.rand(size=(NData, 1, 1))).expand(NData, NVec, DVec))

    Input += Shift

    if type == 'Ranks':
        Output = GetRanks(Input, weights)

    if type == 'Sort':
        Output = GetSorted(Input, weights)

    return Input.to(device), Output.to(device), Shift


