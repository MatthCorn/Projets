import torch

'''
Ce script ajoute des features pour représenter l'état d'une impulsion, et permet aussi de remettre à l'échelle 
les coordonnées à partir de valeurs prédéterminées
'''

class FeaturesAndScaling:

    def __init__(self, divider, weights):
        self.divider = divider
        self.weights = torch.tensor(weights)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, input):
        # input.shape = (batch_size, len_input, 8)
        # parce que l'impulsion contient d'information du type de token
        # on commence par ajouter le temps de fin de l'impulsion
        TOE = input[:, :, 0] + input[:, :, 1]
        # on prend le quotient et le reste de la division de TOE par self.divider
        quotient_TOE, remainder_TOE = TOE.div(self.divider, rounding_mode='trunc'), TOE-TOE.div(self.divider, rounding_mode='trunc')*self.divider
        output = torch.cat((input, quotient_TOE, remainder_TOE), dim=-1)
        return output * self.weights