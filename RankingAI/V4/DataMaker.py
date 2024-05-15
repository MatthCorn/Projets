import torch

def GetSorted(Input, weights):
    if weights is None:
        weights = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, weights.to(Input.device))
    Orders = Values.argsort(dim=-1)
    return Input[torch.arange(Input.size(0)).unsqueeze(1), Orders]

# Les coordonnées intervenants dans l'ordre sont centrées et d'écart type sigma. On applique ensuite une translation sur ces coordonnées.
# L'objectif est de voir la plage de fonctionnement de l'IA sur le tri, surtout voir la conséquence d'une forte translation

def MakeData(NInput=10, NOutput=5, DVec=10, sigma=1, NData=1000, weights=None):
    Input = torch.normal(torch.zeros(NData, NInput, DVec), sigma*torch.ones(NData, NInput, DVec))

    ShiftInterval = [0, 10]
    Shift = ((ShiftInterval[0] + (ShiftInterval[1] - ShiftInterval[0]) * torch.rand(size=(NData, 1, 1))).expand(NData, NInput, DVec))

    Input += Shift

    Output = GetSorted(Input, weights)
    return Input, Output[:, :NOutput]


