import torch
import math

class Hookers():
    def __init__(self, model):
        self.model = model

        self.hookers = []

        self.score_symetry_tracker = 0
        self.score_variance_tracker = 0
        self.score_diversity_decision = 0
        self.score_likelihood_prediction = 0

        self.InitHookers()

    def InitHookers(self):
        self.hookers.append(self.model.prediction_physics.register_forward_hook(self.SymetryTrackers))
        self.hookers.append(self.model.encoder_decider.register_forward_hook(self.DiversityDecision))
        self.hookers.append(self.model.normalizer_decider.register_forward_hook(self.VarianceTrackers))
        self.hookers.append(self.model.normalizer_decider.register_forward_hook(self.BePrediction))

    def RemoveHookers(self, reset=True):
        for hooker in self.hookers:
                hooker.remove()
        if reset:
            self.hookers = []

    def SymetryTrackers(self, module, input, output):
        # on souhaite encourager une sorte de symétrie dans la prédiction des états des mesureurs entre deux instants succéssifs
        # le facteur 2.2 correspond à la distance moyenne entre deux PDWs consécutifs, on permet à l'erreur de valoir cette
        # moyenne afin de ne pas pénaliser la libération de mesureurs entre deux PDWs
        # output.shape = (batch_size, len_target, n_tracker, d_att)
        shape = output.shape
        symetry_layer_score = torch.norm(torch.norm(output[:, 1:] - output[:, :-1])/math.sqrt(shape[0]*(shape[1]-1)*shape[2]*shape[3]) - 2.2)
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

    def BePrediction(self, module, input, output):
        # on souhaite que le vecteur soit du type (1, 0, ..., 0) ou (0, 1, 0, ..., 0) ect -> on l'encourage à être sur les
        # sphères unités de L1 et L2 à la fois (il est déjà sur celle de L1 par construciton)
        # output.shape = (batch_size, len_target, n_tracker, 1)
        likelihood_score = torch.norm(torch.norm(output, dim=2) - 1)
        self.score_likelihood_prediction += likelihood_score

    def ReleaseError(self, type_score='symetry'):
        for name in self.__dict__.keys():
            if (name[:5] == 'score') and (type_score in name):
                break
        temp = self.__dict__[name]
        self.__dict__[name] = 0
        return temp