import torch

'''
Ce script ajoute des features pour représenter l'état d'une impulsion, et permet aussi de remettre à l'échelle 
les coordonnées à partir de valeurs prédéterminées
'''

class FeaturesAndScaling:

    def __init__(self, threshold, freq_ech, weights=None, type='source'):
        # threshold doit être un nombre négatif, plus grand que -(temps de maintien d'un mesureur + temps d'une fenêtre d'impulsions)
        self.threshold = torch.tensor(threshold)
        self.freq_ech = freq_ech
        if weights is None:
            if type == 'source':
                self.weights = torch.tensor([1] * 14, dtype=torch.float32)
            if type == 'target':
                self.weights = torch.tensor([1] * 13, dtype=torch.float32)
        else:
            self.weights = torch.tensor(weights)
        self.type = type

        self.make_mat()

    def __call__(self, x):
        return self.forward(x)

    def make_mat(self):
        if self.type == 'source':

            self.mat_2 = torch.tensor([[1,  0, 0,  0, 0, 0,    0,    0, 0, 0, 0],
                                       [0,  1, 0,  0, 0, 0,    0,    0, 0, 0, 0],
                                       [1, -1, 0,  0, 0, 0,    0,    0, 0, 0, 0],
                                       [0,  0, 1,  0, 0, 0,    0,    0, 0, 0, 0],
                                       [0,  0, 0,  1, 0, 0,    0,    0, 0, 0, 0],
                                       [0,  0, 1, -1, 0, 0,    0,    0, 0, 0, 0],
                                       [0,  0, 0,  0, 1, 0,    0,    0, 0, 0, 0],
                                       [0,  0, 0,  0, 0, 1,    0,    0, 0, 0, 0],
                                       [0,  0, 0,  0, 0, 0,    1,    0, 0, 0, 0],
                                       [0,  0, 0,  0, 0, 0,    0,    1, 0, 0, 0],
                                       [0,  0, 0,  0, 0, 0,  1/2,  1/2, 0, 0, 0],
                                       [0,  0, 0,  0, 0, 0,  1/2,  1/2, 0, 0, 0],
                                       [0,  0, 0,  0, 0, 0, -1/2, -1/2, 0, 0, 0],
                                       [0,  0, 0,  0, 0, 0,  1/2, -1/2, 0, 0, 0],
                                       [0,  0, 0,  0, 0, 0,    0,    0, 1, 0, 0],
                                       [0,  0, 0,  0, 0, 0,    0,    0, 0, 1, 0],
                                       [0,  0, 0,  0, 0, 0,    0,    0, 0, 0, 1]], dtype=torch.float32)

            self.mat_1 = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 0, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)

        if self.type == 'target':

            self.mat_2 = torch.tensor([[1, 0, 0, 0,    0,    0, 0, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0,    0,    0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 1, 0,    0,    0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 1,    0,    0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0,    1,    0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0,    0,    1, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0,  1/2,  1/2, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0,  1/2,  1/2, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, -1/2, -1/2, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0,  1/2, -1/2, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0,    0,    0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0,    0,    0, 0, 1, 0, 0, 0, 0],
                                       [0, 0, 0, 0,    0,    0, 0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 0,    0,    0, 0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0,    0,    0, 0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0,    0,    0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)

            self.mat_1 = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)

    def forward(self, input):
        # on crée le temps de fin d'impulsion (TOE) et on crée 2 copies de TOA et TOE (pour remplacer par le quotien et le reste de la division euclidienne)
        input = torch.matmul(input, self.mat_1.t())

        # on remplace les copies de TO par le quotien de la division euclidienne de TO par self.threshold
        input[..., [1, 3]] = input[..., [1, 3]].div(self.threshold, rounding_mode='floor') + 1

        # on crée le reste de la division euclidienne de TO par self.treshold, Fmoy et 2 copies Fmoy et -Fmoy pour les fréquences repliées et DeltaF
        input = torch.matmul(input, self.mat_2.t())

        # on replie Fmoy et -Fmoy sur [0; self.freq_ech]
        if self.type == 'source':
            input[..., 11:13] = torch.remainder(input[..., 11:13], self.freq_ech)

        if self.type == 'target':
            input[..., 7:9] = torch.remainder(input[..., 7:9], self.freq_ech)

        input[..., :len(self.weights)] *= self.weights

        return input
