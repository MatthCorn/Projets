import torch

'''
Ce script ajoute des features pour représenter l'état d'une impulsion, et permet aussi de remettre à l'échelle 
les coordonnées à partir de valeurs prédéterminées
'''

class FeaturesAndScaling:

    def __init__(self, threshold, freq_ech, weights=None, type='source'):
        # threshold doit être plus grand que (temps de maintien d'un mesureur + temps d'une fenêtre d'impulsions)
        self.threshold = torch.tensor(threshold)
        self.freq_ech = freq_ech
        if weights is None:
            if type == 'source':
                self.weights = torch.tensor([1] * 14, dtype=torch.float32)
            if type == 'target':
                self.weights = torch.tensor([1] * 17, dtype=torch.float32)
        else:
            self.weights = torch.tensor(weights)
        self.type = type
        self.make_mat(type)

    def __call__(self, x):
        return self.forward(x)

    def make_mat(self, type):
        if type == 'source':

            self.mat_1 = torch.tensor([[1, 0, 0,    0,    0, 0, 0, 0],
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

            self.mat_2 = torch.tensor([[1,               0, 0,               0, 0, 0,    0,    0,              0,              0, 0, 0, 0],
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


        if type == 'target':

            self.mat_1 = torch.tensor([[1, 0, 0,    0,    0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 0,    0,    0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 0,    0,    0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 0,    0,    0, 0, 0, 0, 0, 0, 0],
                                       [0, 1, 0,    0,    0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 1,    0,    0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,    1,    0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,    0,    1, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,  1/2,  1/2, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, -1/2, -1/2, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0,    0,    0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0,    0,    0, 0, 1, 0, 0, 0, 0],
                                       [0, 0, 0,    0,    0, 0, 0, 1, 0, 0, 0],
                                       [0, 0, 0,    0,    0, 0, 0, 0, 1, 0, 0],
                                       [0, 0, 0,    0,    0, 0, 0, 0, 0, 1, 0],
                                       [0, 0, 0,    0,    0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)

            self.mat_2 = torch.tensor([[1,               0, 0,               0, 0, 0,    0,    0,              0,              0, 0, 0, 0, 0, 0, 0],
                                       [0,               1, 0,               0, 0, 0,    0,    0,              0,              0, 0, 0, 0, 0, 0, 0],
                                       [1, -self.threshold, 0,               0, 0, 0,    0,    0,              0,              0, 0, 0, 0, 0, 0, 0],
                                       [0,               0, 1,               0, 0, 0,    0,    0,              0,              0, 0, 0, 0, 0, 0, 0],
                                       [0,               0, 0,               1, 0, 0,    0,    0,              0,              0, 0, 0, 0, 0, 0, 0],
                                       [0,               0, 1, -self.threshold, 0, 0,    0,    0,              0,              0, 0, 0, 0, 0, 0, 0],
                                       [0,               0, 0,               0, 1, 0,    0,    0,              0,              0, 0, 0, 0, 0, 0, 0],
                                       [0,               0, 0,               0, 0, 1,    0,    0,              0,              0, 0, 0, 0, 0, 0, 0],
                                       [0,               0, 0,               0, 0, 0,    1,    0,              0,              0, 0, 0, 0, 0, 0, 0],
                                       [0,               0, 0,               0, 0, 0,    0,    1,              0,              0, 0, 0, 0, 0, 0, 0],
                                       [0,               0, 0,               0, 0, 0,  1/2,  1/2,              0,              0, 0, 0, 0, 0, 0, 0],
                                       [0,               0, 0,               0, 0, 0,  1/2,  1/2, -self.freq_ech,              0, 0, 0, 0, 0, 0, 0],
                                       [0,               0, 0,               0, 0, 0, -1/2, -1/2,              0, -self.freq_ech, 0, 0, 0, 0, 0, 0],
                                       [0,               0, 0,               0, 0, 0,  1/2, -1/2,              0,              0, 0, 0, 0, 0, 0, 0],
                                       [0,               0, 0,               0, 0, 0,    0,    0,              0,              0, 1, 0, 0, 0, 0, 0],
                                       [0,               0, 0,               0, 0, 0,    0,    0,              0,              0, 0, 1, 0, 0, 0, 0],
                                       [0,               0, 0,               0, 0, 0,    0,    0,              0,              0, 0, 0, 1, 0, 0, 0],
                                       [0,               0, 0,               0, 0, 0,    0,    0,              0,              0, 0, 0, 0, 1, 0, 0],
                                       [0,               0, 0,               0, 0, 0,    0,    0,              0,              0, 0, 0, 0, 0, 1, 0],
                                       [0,               0, 0,               0, 0, 0,    0,    0,              0,              0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)



    def forward(self, input):
        # on crée le temps de fin d'impulsion (TOE) et on crée 1 copie de TOA et TOE (pour remplacer par le quotien de la division euclidienne)
        # on crée Fmoy et -Fmoy pour les fréquences repliées
        input = torch.matmul(input, self.mat_1.t())

        # on remplace la copie des TO par leurs quotiens par la division euclidienne avec self.threshold
        # on remplace Fmoy et -Fmoy par leurs quotiens par la division euclidienne avec self.freq_ech
        input[..., [1, 3, 8, 9]] = input[..., [1, 3, 8, 9]].div(torch.tensor([self.threshold, self.threshold, self.freq_ech, self.freq_ech]), rounding_mode='floor') + \
                                   torch.tensor([1, 1, 0, 0])

        # on crée le reste de la division euclidienne des TO par self.threshold (on a déjà le quotien)
        # on remplace le quotien de la division euclidienne de Fmoy et -Fmoy par self.freq_ech par leurs restent
        # on crée Fmoy et DeltaF
        input = torch.matmul(input, self.mat_2.t())


        input[..., :len(self.weights)] *= self.weights

        return input
