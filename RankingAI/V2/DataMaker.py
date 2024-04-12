import torch
from Complete.Transformer.PositionalEncoding import PositionalEncoding

def GetRanks(Input, weights):
    if weights is None:
        weights = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, weights.to(Input.device))
    Orders = Values.argsort(dim=-1)
    return Orders

# Cette fonction permet de créer des données où la position initiale des vecteurs dans la séquence et la position des vecteurs une fois triés
# sont fournises (concaténées) directement dans le vecteur en entrée de l'encodeur du transformer

def CheatedData(NVec=5, DVec=10, sigma=1, NData=1000, ShiftInterval=[0, 100], d_att=64, weights=None):
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


