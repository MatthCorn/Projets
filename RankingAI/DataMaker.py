import torch
from Complete.Transformer.PositionalEncoding import PositionalEncoding

def GetRanks(Input, weights):
    if weights is None:
        weights = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, weights.to(Input.device))
    Orders = Values.argsort(dim=-1)
    return Orders

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
        Output = GetRanks(Input, weights).unsqueeze(dim=-1)
        return Input.to(device), Shift, Output.to(device)

    if type == 'Sort':
        Output = GetSorted(Input, weights)
        return Input.to(device), Shift, Output.to(device)

    Ranks = GetRanks(Input, weights)
    Sort = GetSorted(Input, weights)
    return Input.to(device), Shift, Ranks.to(device), Sort.to(device)

# Cette fonction permet de créer des données où la position initiale des vecteurs dans la séquence et la position des vecteurs une fois triés
# sont fournises (concaténées) directement dans le vecteur en entrée de l'encodeur du transformer

def CheatedData(NVec=5, DVec=10, sigma=1, NData=1000, ShiftInterval=[0, 100], d_att=64, weights=None, device=torch.device('cpu')):
    Input = torch.normal(torch.zeros(NData, NVec, DVec), sigma*torch.ones(NData, NVec, DVec))

    Shift = ((ShiftInterval[0] + (ShiftInterval[1] - ShiftInterval[0])*torch.rand(size=(NData, 1, 1))).expand(NData, NVec, DVec))

    Input += Shift

    Ranks = GetRanks(Input, weights)

    PosEnc = PositionalEncoding(d_att=d_att, max_len=NVec)

    empty = torch.zeros(NData, NVec, d_att)

    InitialPE = PosEnc(empty)
    FinalPE = PosEnc(empty)[torch.arange(Input.size(0)).unsqueeze(1), Ranks]

    Output = Input[torch.arange(Input.size(0)).unsqueeze(1), Ranks]

    return Input, InitialPE, FinalPE, Output, Ranks


