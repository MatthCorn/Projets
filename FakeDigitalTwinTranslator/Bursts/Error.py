import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from math import sqrt
import numpy as np

# Pond = [1., 40, 3, 4, 3, 0.2, 0.2, 0.2]
ActPond = 10
Pond = [1.]*8
# ActPond = 1
# Ce script définit les erreurs utilisées pour l'appentissage et pour visualisé si l'objectif final est atteint

def TrainingError(Source, Translation, Ended, batch_size, batch_indice, Translator):
    j = batch_indice
    BatchSource = Source[j * batch_size: (j + 1) * batch_size].detach().to(device=Translator.device, dtype=torch.float32)
    BatchTranslation = Translation[j * batch_size: (j + 1) * batch_size].detach().to(device=Translator.device, dtype=torch.float32)
    BatchEnded = Ended[j * batch_size: (j + 1) * batch_size].detach().to(device=Translator.device, dtype=torch.float32)
    # BatchEnded.shape = (batch_size, latent_len, 1)

    # Pour faire en sorte que l'action "continuer d'écrire" soit autant représentée en True qu'en False, on pondère le masque des actions
    NumWords = torch.sum((1 - BatchEnded), dim=1).unsqueeze(-1).to(torch.int32)
    # NumWords.shape = (batch_size, 1, 1)
    BatchActionMask = F.pad(1 - BatchEnded, [0, 0, 0, 1])/torch.sqrt((2*NumWords + 1e-5))
    for i in range(len(NumWords)):
        if int(NumWords[i]) == 0:
            BatchActionMask[(i, int(NumWords[i]))] = 1
        else:
            BatchActionMask[(i, int(NumWords[i]))] = 1/4

    PredictedTranslation, PredictedAction = Translator.forward(source=BatchSource, target=BatchTranslation)

    # Comme la décision d'arrêter d'écrire passe par Action, et pas par un token <end>, on ne s'interesse pas au dernier mot. En revenche, la dernière action,
    # si observée, ne peut être que "attendre la nouvelle salve", sinon cela signifie que le réseau est la dimensionné (target_len - NbPDWsMemory trop petit).
    # On ne s'interesse aussi qu'aux prédictions à partir de la position "Translator.NbPDWsMemory", ce qui correspond aux prédictions devant être
    # faites à l'arrivée de la dernière salve.

    PredictedTranslation = PredictedTranslation[:, Translator.NbPDWsMemory-1:-1]
    # PredictedTranslation.shape = (batch_size, target_len-PDWsMemory, d_target+num_flags)
    PredictedAction = PredictedAction[:, Translator.NbPDWsMemory-1:]
    # PredictedAction.shape = (batch_size, target_len-PDWsMemory+1, 1)

    ErrTrans = torch.norm((BatchTranslation[:, Translator.NbPDWsMemory:] - PredictedTranslation) * (1 - BatchEnded), dim=(0, 1))/float(torch.norm((1 - BatchEnded)))
    # ErrTrans.shape = d_target+num_flags
    ErrAct = torch.norm((PredictedAction - F.pad(1 - BatchEnded, [0, 0, 0, 1])) * BatchActionMask)/sqrt(batch_size)

    return ErrTrans, ErrAct

def ObjectiveError(Source, Translation, Ended, Translator):
    PredictedTranslation = Translator.translate(Source.to(device=Translator.device))
    PredictedTranslation = pad_sequence(PredictedTranslation, batch_first=True).to(device=Translator.device)

    Translation, Ended = Translation.to(device=Translator.device), Ended.to(device=Translator.device)


    _, temp_len, _ = PredictedTranslation.shape
    _, len_target, d_out = Translation.shape

    if temp_len < len_target:
        PredictedTranslation = F.pad(PredictedTranslation, (0, 0, 0, len_target - temp_len))
    if temp_len > len_target:
        Translation = F.pad(Translation, (0, 0, 0, temp_len - len_target))
        Ended = F.pad(Ended, (0, 0, 0, temp_len - len_target))

    # On calcule l'erreur de la phrase intentionnellement écrite.
    # C'est-à-dire qu'on prend en compte la fin de l'écriture gérée par Action et représentée par le masque PredictedMaskTranslation
    RealError = torch.norm(PredictedTranslation - Translation)/float(torch.norm((1 - Ended)))

    # On calcule l'erreur sur le même nombre de mot que les traductions attendues
    CutError = torch.norm(PredictedTranslation * (1-Ended) - Translation)/float(torch.norm((1 - Ended)))

    return float(RealError)/sqrt(d_out), float(CutError)/sqrt(d_out)

def ErrorAction(Source, Translation, Ended, Translator, batch_size=50, Action='', Optimizer=None):
    data_size, _, d_out = Translation.shape
    n_batch = int(data_size/batch_size)

    if Action == 'Evaluation':
        with torch.no_grad():
            RealError, CutError = ObjectiveError(Source, Translation, Ended, Translator)
        return RealError, CutError


    Error = 0
    ErrTrans = []
    ErrAct = 0
    for j in range(n_batch):
        err = 0
        if Action == 'Training':
            Optimizer.zero_grad(set_to_none=True)

            errtrans, erract = TrainingError(Source, Translation, Ended, batch_size, j, Translator)
            err = torch.sqrt(torch.norm(errtrans * torch.tensor(Pond, device=Translator.device))**2/(4*d_out) + (ActPond*erract)**2/4)
            err.backward()
            Optimizer.step()
        elif Action == 'Validation':
            with torch.no_grad():
                errtrans, erract = TrainingError(Source, Translation, Ended, batch_size, j, Translator)
                err = torch.sqrt(torch.norm(errtrans * torch.tensor(Pond, device=Translator.device))**2/(4*d_out) + (ActPond*erract)**2/4)

        Error += float(err)**2
        ErrAct += float(erract)**2
        ErrTrans.append(errtrans**2)
    ErrTrans = torch.sqrt(sum(ErrTrans)/n_batch).tolist()
    Error = sqrt(Error/n_batch)
    ErrAct = sqrt(ErrAct/n_batch)
    return Error, ErrAct, ErrTrans

# Cette fonction permet de savoir l'évolution de l'erreur générale ou d'une coordonnée en fonction de la position
# dans la phrase traduite (sachant qu'on s'attend à voir une propagation de l'erreur au fil de la traduction)
def DetailObjectiveError(Source, Translation, Ended, Translator, dim=None):
    NbTranslation = len(Source)
    PredictedTranslation = Translator.translate(Source.to(device=Translator.device))

    NumWords = torch.sum((1 - Ended), dim=1).squeeze(-1).to(torch.int)
    Translation = [Translation[i][:NumWords[i]] for i in range(NbTranslation)]

    CutDiffList = []
    for i in range(NbTranslation):
        PredTrans = PredictedTranslation[i]
        Trans = Translation[i]
        if len(Trans) > len(PredTrans):
            PredTrans = F.pad(PredTrans, (0, 0, 0, len(Trans) - len(PredTrans)))
        elif len(Trans) < len(PredTrans):
            PredTrans = PredTrans[:len(Trans)]


    RealDiffList = []

    _, temp_len, _ = PredictedTranslation.shape
    _, len_target, d_out = Translation.shape

    if temp_len < len_target:
        PredictedTranslation = F.pad(PredictedTranslation, (0, 0, 0, len_target - temp_len))
    if temp_len > len_target:
        Translation = F.pad(Translation, (0, 0, 0, temp_len - len_target))
        Ended = F.pad(Ended, (0, 0, 0, temp_len - len_target))

    # On calcule l'erreur de la phrase intentionnellement écrite.
    # C'est-à-dire qu'on prend en compte la fin de l'écriture gérée par Action et représentée par le masque PredictedMaskTranslation
    RealDiff = (PredictedTranslation - Translation)/float(torch.norm((1 - Ended)))

    np.polyfit(Shift, ErrShift, deg=4)

    # On calcule l'erreur sur le même nombre de mot que les traductions attendues
    CutDiff = torch.norm(PredictedTranslation * (1-Ended) - Translation)/float(torch.norm((1 - Ended)))

    return float(RealError)/sqrt(d_out), float(CutError)/sqrt(d_out)