import torch

# Ce script définit les erreurs utilisées pour l'appentissage et pour visualisé si l'objectif final est atteint

def TrainingError(Source, Translation, Ended, batch_size, batch_indice, Translator):
    j = batch_indice
    BatchSource = Source[j * batch_size: (j + 1) * batch_size].detach().to(device=Translator.device, dtype=torch.float32)
    BatchTranslation = Translation[j * batch_size: (j + 1) * batch_size].detach().to(device=Translator.device, dtype=torch.float32)
    BatchEnded = Ended[j * batch_size: (j + 1) * batch_size].detach().to(device=Translator.device, dtype=torch.float32)

    # Pour faire en sorte que l'action "continuer d'écrire" soit autant représentée en True qu'en False, on pondère le masque des actions
    NumWords = torch.sum((1 - BatchEnded), dim=1).unsqueeze(-1).to(torch.int32)
    BatchActionMask = (1 - BatchEnded) / NumWords
    for i in range(len(NumWords)):
        BatchActionMask[(i, int(NumWords[i]))] = 2

    PredictedTranslation, PredictedAction = Translator.forward(source=BatchSource, target=BatchTranslation)

    # Comme la décision d'arrêter d'écrire passe par Action, et pas par un token <end>, on ne s'interesse pas au dernier mot
    # On ne s'interesse pas non plus à l'action prédite par le token <start>, puisque de toute manière il est suivi d'un mot
    PredictedTranslation = PredictedTranslation[:, 1:]
    PredictedAction = PredictedAction[:, :-1]
    err = (torch.norm((BatchTranslation - PredictedTranslation) * (1 - BatchEnded)) +
           torch.norm((PredictedAction - (1 - BatchEnded)) * BatchActionMask)) / batch_size

    return err

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
    for j in range(n_batch):
        err = 0
        if Action == 'Training':
            Optimizer.zero_grad(set_to_none=True)

            err = TrainingError(Source, Translation, Ended, batch_size, j, Translator)
            err.backward()
            Optimizer.step()
        elif Action == 'Validation':
            with torch.no_grad():
                err = TrainingError(Source, Translation, Ended, batch_size, j, Translator)
        Error += float(err)*batch_size
    return Error/data_size