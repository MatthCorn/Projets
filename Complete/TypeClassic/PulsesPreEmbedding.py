import torch

'''
Ce script ajoute des features pour représenter l'état d'une impulsion, et permet aussi de remettre à l'échelle 
les coordonnées à partir de valeurs prédéterminées
'''

class FeaturesAndScaling:

    def __init__(self, threshold, weights=[1, 1, 1, 1, 1, 1, 1, 1]):
        # threshold doit être un nombre négatif, plus grand que -(temps de maintien d'un mesureur + temps d'une fenêtre d'impulsions)
        self.threshold = torch.tensor(threshold)
        self.weights = torch.tensor(weights)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, input):
        # input.shape = (batch_size, len_input, 8)
        # parce que l'impulsion contient d'information du type de token
        # on commence par ajouter le temps de fin de l'impulsion
        TOE = input[:, :, 0] + input[:, :, 1]
        # on prend le quotient et le reste de la division de TOE par self.divider
        acc_TOE, unacc_TOE = torch.max(TOE, self.threshold), torch.min(torch.zeros(1), TOE - self.threshold)
        output = torch.cat((input[:, :, :2], TOE, acc_TOE, unacc_TOE, input[:, :, 2:]), dim=-1)
        return output * self.weights
