import torch
from math import sqrt


# Ce script définit les erreurs utilisées pour l'appentissage et pour visualisé si l'objectif final est atteint

def TrainingError(Source, Translation, batch_size, batch_indice, Translator):
    j = batch_indice
    BatchSource = Source[j * batch_size: (j + 1) * batch_size].detach().to(device=Translator.device, dtype=torch.float32)
    BatchTranslation = Translation[j * batch_size: (j + 1) * batch_size].detach().to(device=Translator.device, dtype=torch.float32)

    PredictedTranslation = Translator.forward(source=BatchSource, target=BatchTranslation)

    shape = BatchTranslation.shape
    ErrTrans = torch.norm(BatchTranslation - PredictedTranslation, dim=(0, 1))/sqrt(shape[0]*shape[1])

    return ErrTrans


def ErrorAction(Source, Translation, Translator, batch_size=50, Action='', Optimizer=None):
    data_size, _, d_out = Translation.shape
    n_batch = int(data_size/batch_size)

    Error = 0
    ErrTrans = []
    for j in range(n_batch):
        err = 0
        if Action == 'Training':
            Optimizer.zero_grad(set_to_none=True)

            ErrTrans = TrainingError(Source, Translation, batch_size, j, Translator)
            err = torch.norm(ErrTrans)/sqrt(d_out)
            err.backward()
            Optimizer.step()

        elif Action == 'Validation':
            with torch.no_grad():
                ErrTrans = TrainingError(Source, Translation, batch_size, j, Translator)
                err = torch.norm(ErrTrans) / sqrt(d_out)

        Error += float(err)**2
        ErrTrans.append(ErrTrans**2)

    ErrTrans = torch.sqrt(sum(ErrTrans)/n_batch).tolist()
    Error = sqrt(Error/n_batch)

    return Error, ErrTrans
