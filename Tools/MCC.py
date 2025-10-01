import torch

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
    best_thr = sorted_scores[best_idx].item()

    return best_thr, best_mcc