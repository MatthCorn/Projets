import torch
import torch.nn as nn
import torch.nn.functional as F


def best_mcc_threshold_torch(y_true, y_score):
    """
    Trouve le seuil qui maximise le MCC
    y_true : tensor (N,) contenant {0,1}
    y_score : tensor (N,) contenant les scores (probabilités)
    """
    # trier par score décroissant
    sorted_scores, order = torch.sort(y_score, descending=True)
    sorted_true = y_true[order]

    # totaux
    P = sorted_true.sum()
    N = len(sorted_true) - P

    # cumul des TP et FP
    tp = torch.cumsum(sorted_true, dim=0)
    fp = torch.cumsum(1 - sorted_true, dim=0)

    # FN et TN déduits
    fn = P - tp
    tn = N - fp

    # MCC numérateur et dénominateur
    numerator = tp * tn - fp * fn
    denominator = torch.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )

    # éviter les divisions par 0
    mcc = torch.zeros_like(numerator, dtype=torch.float)
    valid = denominator > 0
    mcc[valid] = numerator[valid].float() / denominator[valid]

    # trouver MCC max
    best_idx = torch.argmax(mcc)
    best_mcc = mcc[best_idx].item()

    # seuil = milieu entre deux scores (plus robuste que prendre score exact)
    if best_idx < len(sorted_scores) - 1:
        best_thr = (sorted_scores[best_idx] + sorted_scores[best_idx+1]) / 2
    else:
        # cas particulier : tout est positif, seuil = min score
        best_thr = sorted_scores[best_idx]

    return float(best_thr), best_mcc


def soft_mcc(probs, targets, mask, eps=1e-5):
    # Calcul des composantes "Soft"
    tp = torch.sum(targets * probs * mask, dim=[1, 2], keepdim=True)
    tn = torch.sum((1 - targets) * (1 - probs) * mask, dim=[1, 2], keepdim=True)
    fp = torch.sum((1 - targets) * probs * mask, dim=[1, 2], keepdim=True)
    fn = torch.sum(targets * (1 - probs) * mask, dim=[1, 2], keepdim=True)

    # Formule du MCC
    numerator = (tp * tn) - (fp * fn)
    denominator = torch.sqrt(
        (tp + fp + eps) * (tp + fn + eps) * (tn + fp + eps) * (tn + fn + eps)
    )

    mcc = numerator / denominator

    return mcc


class CosineDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Cette couche Linear(1, 1) sert à apprendre :
        # 1. La Température (pente) : pour étirer le cosinus [-1, 1] vers -inf/+inf
        # 2. Le Biais (seuil) : pour décaler le point de bascule (0.5)
        self.scaler = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature, reference):
        # 1. Calcul de la similarité (Réduction de dimension)
        # feature: [Batch, Seq, Dim], reference: [1, 1, Dim] -> out: [Batch, Seq]
        sim = F.cosine_similarity(feature, reference, dim=-1)

        # 2. On remet une dimension pour que le Linear l'accepte
        # out: [Batch, Seq, 1]
        sim = sim.unsqueeze(-1)

        # 3. Scaling + Sigmoid
        # La couche Linear transforme la similarité brute en "logit"
        return self.sigmoid(self.scaler(sim))

