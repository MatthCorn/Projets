import torch
import torch.nn.functional as F

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
    BatchActionMask = F.pad(1 - BatchEnded, [0, 0, 0, 1])/ (NumWords + 1e-5)
    for i in range(len(NumWords)):
        BatchActionMask[(i, int(NumWords[i]))] = 1

    PredictedTranslation, PredictedAction = Translator.forward(source=BatchSource, target=BatchTranslation)

    # Comme la décision d'arrêter d'écrire passe par Action, et pas par un token <end>, on ne s'interesse pas au dernier mot. En revenche, la dernière action,
    # si observée, ne peut être que "attendre la nouvelle salve", sinon cela signifie que le réseau est la dimensionné (target_len - NbPDWsMemory trop petit).
    # On ne s'interesse aussi qu'aux prédictions à partir de la position "Translator.NbPDWsMemory", ce qui correspond aux prédictions devant être
    # faites à l'arrivée de la dernière salve.

    PredictedTranslation = PredictedTranslation[:, Translator.NbPDWsMemory-1:-1]
    # PredictedTranslation.shape = (batch_size, target_len-PDWsMemory, d_target+num_flags)
    PredictedAction = PredictedAction[:, Translator.NbPDWsMemory-1:]
    # PredictedAction.shape = (batch_size, target_len-PDWsMemory+1, 1)

    ErrTrans = torch.norm((BatchTranslation[:, Translator.NbPDWsMemory:] - PredictedTranslation) * (1 - BatchEnded) / (NumWords + 1e-5), dim=(0, 1))/batch_size
    # ErrTrans.shape = d_target+num_flags
    ErrAct = torch.norm((PredictedAction - F.pad(1 - BatchEnded, [0, 0, 0, 1])) * BatchActionMask)/batch_size

    return ErrTrans, ErrAct

def ObjectiveError(Source, Translation, Ended, Translator):
    PredictedTranslation, PredictedMaskTranslation = Translator.translate(Source.to(device=Translator.device))
    Translation, Ended = Translation.to(device=Translator.device), Ended.to(device=Translator.device)

    # On calcule l'erreur de la phrase intentionnellement écrite.
    # C'est-à-dire qu'on prend en compte la fin de l'écriture gérée par Action et représentée par le masque PredictedMaskTranslation
    RealError = torch.norm(PredictedTranslation * PredictedMaskTranslation - Translation)

    # On calcule l'erreur sur le même nombre de mot que les traductions attendues
    CutError = torch.norm(PredictedTranslation * (1-Ended) - Translation)

    return float(RealError)/len(Source), float(CutError)/len(Source)


def ErrorAction(Source, Translation, Ended, Translator, batch_size=50, Action='', Optimizer=None):
    data_size = len(Source)
    n_batch = int(data_size/batch_size)

    if Action == 'Evaluation':
        with torch.no_grad():
            RealError, CutError = ObjectiveError(Source, Translation, Ended, Translator)
        return RealError/data_size, CutError/data_size


    Error = 0
    ErrTrans = []
    ErrAct = 0
    for j in range(n_batch):
        err = 0
        if Action == 'Training':
            Optimizer.zero_grad(set_to_none=True)

            errtrans, erract = TrainingError(Source, Translation, Ended, batch_size, j, Translator)
            err = torch.norm(errtrans * torch.tensor([1., 40, 3, 4, 3, 0.2, 0.2, 0.2], device=Translator.device)) + 40 * erract
            err.backward()
            Optimizer.step()
        elif Action == 'Validation':
            with torch.no_grad():
                errtrans, erract = TrainingError(Source, Translation, Ended, batch_size, j, Translator)
                err = torch.norm(errtrans * torch.tensor([1., 40, 3, 4, 3, 0.2, 0.2, 0.2], device=Translator.device)) + 40 * erract

        Error += float(err/n_batch)
        ErrAct += float(erract/n_batch)
        ErrTrans.append(errtrans/n_batch)
    ErrTrans = sum(ErrTrans).tolist()
    return Error, ErrAct, ErrTrans