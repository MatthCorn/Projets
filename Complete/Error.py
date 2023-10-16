import torch
from torch.nn import functional as F
import math

class Hookers():
    def __init__(self, model):
        self.model = model

        self.hookers = []

        self.score_symetry_tracker = 0
        self.score_variance_tracker = 0
        self.score_diversity_decision = 0

        self.InitHookers()

    def InitHookers(self):
        for decoder in self.model.decoders:
            self.hookers.append(decoder.register_forward_hook(self.SymetryTrackers))
        self.hookers.append(self.model.encoder_decider.register_forward_hook(self.DiversityDecision))
        self.hookers.append(self.model.normalizer_decider.register_forward_hook(self.VarianceTrackers))

    def RemoveHookers(self, reset=True):
        for hooker in self.hookers:
                hooker.remove()
        if reset:
            self.hookers = []

    def SymetryTrackers(self, module, input, output):
        # on souhaite encourager une sorte de symétrie dans la prédiction des états des mesureurs entre deux instants succéssifs
        # output.shape = (batch_size, n_tracker*len_target, d_att)
        shape = output.shape
        symetry_layer_score = torch.norm(output[:, self.model.n_tracker:] - output[:, :-self.model.n_tracker])/math.sqrt(shape[0]*shape[1]*shape[2])
        self.score_symetry_tracker += symetry_layer_score

    def VarianceTrackers(self, module, input, output):
        # on souhaite favoriser la diversité dans les impulsions suivis par les mesureurs à chaque instant
        input = input[0]
        # input.shape = (batch_size*len_target, n_tracker, d_att)
        shape = input.shape
        variance_score = torch.norm(torch.var(input, dim=1))/math.sqrt(shape[0]*shape[2])
        self.score_variance_tracker += variance_score

    def DiversityDecision(self, module, input, output):
        # on souhaite guider l'IA sur le fait que tous les mesureurs sont libérés environ aussi souvent lors d'un scénario
        # output.shape = (batch_size, len_target, n_tracker, 1)
        diversity_score = torch.norm(torch.var(output, dim=(0, 1)))/math.sqrt(self.model.n_tracker)
        self.score_diversity_decision += diversity_score

    def ReleaseError(self, type_score='symetry'):
        for name in self.__dict__.keys():
            if (name[:5] == 'score') and (type_score in name):
                break
        temp = self.__dict__[name]
        self.__dict__[name] = 0
        return temp

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


def ErrorAction(Source, Translation, Ended, Translator, Hookkers, batch_size=50, Action='', Optimizer=None):
    data_size, _, d_out = Translation.shape
    n_batch = int(data_size/batch_size)

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
    Error = math.sqrt(Error/n_batch)
    ErrAct = math.sqrt(ErrAct/n_batch)
    return Error, ErrAct, ErrTrans