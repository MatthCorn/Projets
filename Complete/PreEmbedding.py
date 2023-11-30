import torch
import numpy as np
import os

'''
Ce script ajoute des features pour représenter l'état d'une impulsion, et permet aussi de remettre à l'échelle 
les coordonnées à partir de valeurs prédéterminées
'''

class FeaturesAndScaling(torch.nn.Module):

    def __init__(self, threshold, freq_ech, weights=None, type='source'):
        super().__init__()
        # threshold doit être plus grand que (temps de maintien d'un mesureur + temps d'une fenêtre d'impulsions)
        self.threshold = threshold
        self.freq_ech = freq_ech
        self.type = type
        if weights is None:
            if type == 'source':
                self.register_buffer('average', torch.tensor([0] * 14, dtype=torch.float32), persistent=False)
                self.register_buffer('std', torch.tensor([1] * 14, dtype=torch.float32), persistent=False)
            if type == 'target':
                self.register_buffer('average', torch.tensor([0] * 17, dtype=torch.float32), persistent=False)
                self.register_buffer('std', torch.tensor([1] * 17, dtype=torch.float32), persistent=False)
        else:
            self.LoadWeights(weights)
        self.make_mat(type)

    def make_mat(self, type):
        if type == 'source':

            mat_1 = torch.tensor([[1, 0, 0,    0,    0],
                                  [1, 0, 0,    0,    0],
                                  [1, 1, 0,    0,    0],
                                  [1, 1, 0,    0,    0],
                                  [0, 1, 0,    0,    0],
                                  [0, 0, 1,    0,    0],
                                  [0, 0, 0,    1,    0],
                                  [0, 0, 0,    0,    1],
                                  [0, 0, 0,  1/2,  1/2],
                                  [0, 0, 0, -1/2, -1/2]], dtype=torch.float32)

            mat_2 = torch.tensor([[1,               0, 0,               0, 0, 0,    0,    0,              0,              0],
                                  [0,               1, 0,               0, 0, 0,    0,    0,              0,              0],
                                  [1, -self.threshold, 0,               0, 0, 0,    0,    0,              0,              0],
                                  [0,               0, 1,               0, 0, 0,    0,    0,              0,              0],
                                  [0,               0, 0,               1, 0, 0,    0,    0,              0,              0],
                                  [0,               0, 1, -self.threshold, 0, 0,    0,    0,              0,              0],
                                  [0,               0, 0,               0, 1, 0,    0,    0,              0,              0],
                                  [0,               0, 0,               0, 0, 1,    0,    0,              0,              0],
                                  [0,               0, 0,               0, 0, 0,    1,    0,              0,              0],
                                  [0,               0, 0,               0, 0, 0,    0,    1,              0,              0],
                                  [0,               0, 0,               0, 0, 0,  1/2,  1/2,              0,              0],
                                  [0,               0, 0,               0, 0, 0,  1/2,  1/2, -self.freq_ech,              0],
                                  [0,               0, 0,               0, 0, 0, -1/2, -1/2,              0, -self.freq_ech],
                                  [0,               0, 0,               0, 0, 0,  1/2, -1/2,              0,              0]], dtype=torch.float32)


        if type == 'target':

            mat_1 = torch.tensor([[1, 0, 0,    0,    0, 0, 0, 0],
                                  [1, 0, 0,    0,    0, 0, 0, 0],
                                  [1, 1, 0,    0,    0, 0, 0, 0],
                                  [1, 1, 0,    0,    0, 0, 0, 0],
                                  [0, 1, 0,    0,    0, 0, 0, 0],
                                  [0, 0, 1,    0,    0, 0, 0, 0],
                                  [0, 0, 0,    1,    0, 0, 0, 0],
                                  [0, 0, 0,    0,    1, 0, 0, 0],
                                  [0, 0, 0,  1/2,  1/2, 0, 0, 0],
                                  [0, 0, 0, -1/2, -1/2, 0, 0, 0],
                                  [0, 0, 0,    0,    0, 1, 0, 0],
                                  [0, 0, 0,    0,    0, 0, 1, 0],
                                  [0, 0, 0,    0,    0, 0, 0, 1]], dtype=torch.float32)

            mat_2 = torch.tensor([[1,               0, 0,               0, 0, 0,    0,    0,              0,              0, 0, 0, 0],
                                  [0,               1, 0,               0, 0, 0,    0,    0,              0,              0, 0, 0, 0],
                                  [1, -self.threshold, 0,               0, 0, 0,    0,    0,              0,              0, 0, 0, 0],
                                  [0,               0, 1,               0, 0, 0,    0,    0,              0,              0, 0, 0, 0],
                                  [0,               0, 0,               1, 0, 0,    0,    0,              0,              0, 0, 0, 0],
                                  [0,               0, 1, -self.threshold, 0, 0,    0,    0,              0,              0, 0, 0, 0],
                                  [0,               0, 0,               0, 1, 0,    0,    0,              0,              0, 0, 0, 0],
                                  [0,               0, 0,               0, 0, 1,    0,    0,              0,              0, 0, 0, 0],
                                  [0,               0, 0,               0, 0, 0,    1,    0,              0,              0, 0, 0, 0],
                                  [0,               0, 0,               0, 0, 0,    0,    1,              0,              0, 0, 0, 0],
                                  [0,               0, 0,               0, 0, 0,  1/2,  1/2,              0,              0, 0, 0, 0],
                                  [0,               0, 0,               0, 0, 0,  1/2,  1/2, -self.freq_ech,              0, 0, 0, 0],
                                  [0,               0, 0,               0, 0, 0, -1/2, -1/2,              0, -self.freq_ech, 0, 0, 0],
                                  [0,               0, 0,               0, 0, 0,  1/2, -1/2,              0,              0, 0, 0, 0],
                                  [0,               0, 0,               0, 0, 0,    0,    0,              0,              0, 1, 0, 0],
                                  [0,               0, 0,               0, 0, 0,    0,    0,              0,              0, 0, 1, 0],
                                  [0,               0, 0,               0, 0, 0,    0,    0,              0,              0, 0, 0, 1]], dtype=torch.float32)

        self.register_buffer('mat_1', mat_1, persistent=False)
        self.register_buffer('mat_2', mat_2, persistent=False)
        self.register_buffer('divider', torch.tensor([self.threshold, self.threshold, self.freq_ech, self.freq_ech]), persistent=False)
        self.register_buffer('const', torch.tensor([1, 1, 0, 0]), persistent=False)

    def LoadWeights(self, path):
        self.register_buffer('average', torch.tensor(np.load(os.path.join(path, self.type + '_average.npy'))))
        self.register_buffer('std', torch.tensor(np.load(os.path.join(path, self.type + '_std.npy')) + 1e-10))

    def forward(self, input):
        # on crée le temps de fin d'impulsion (TOE) et on crée 1 copie de TOA et TOE (pour remplacer par le quotien de la division euclidienne)
        # on crée Fmoy et -Fmoy pour les fréquences repliées
        input = torch.matmul(input, self.mat_1.t())

        # on remplace la copie des TO par leurs quotiens par la division euclidienne avec self.threshold
        # on remplace Fmoy et -Fmoy par leurs quotiens par la division euclidienne avec self.freq_ech
        input[..., [1, 3, 8, 9]] = input[..., [1, 3, 8, 9]].div(self.divider, rounding_mode='floor') + self.const

        # on crée le reste de la division euclidienne des TO par self.threshold (on a déjà le quotien)
        # on remplace le quotien de la division euclidienne de Fmoy et -Fmoy par self.freq_ech par leurs restent
        # on crée Fmoy et DeltaF
        input = torch.matmul(input, self.mat_2.t())

        input = (input - self.average) / self.std

        return input
