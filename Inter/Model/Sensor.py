import torch
from torch.nn.functional import pad

class Simulator:
    def __init__(self, dim, sensitivity, n_sat, n_mes=6, WeightF=None, WeightL=None, model_path=None):
        self.dim = dim
        self.sensitivity = sensitivity
        self.n_sat = n_sat
        self.n_mes = n_mes
        self.weight_f = torch.tensor(WeightF, dtype=torch.float) if WeightF is not None else torch.tensor([1., 0.] + [0.] * (self.dim - 2))
        self.weight_f = self.weight_f / torch.norm(self.weight_f)
        self.weight_l = torch.tensor(WeightL, dtype=torch.float) if WeightL is not None else torch.tensor([0., 1.] + [0.] * (self.dim - 2))
        self.weight_l = self.weight_l / torch.norm(self.weight_l)
        self.P = []     # contient les vecteurs de suivis, qui enregistrent itérativement les informations des vecteurs présents
        self.TM = []    # contient le temps de dernière mise-à-jour de chaque vecteur de suivi
        self.TI = []    # contient le temps d'apparition de chaque vecteur de suivi
        self.R = []     # contient tous les vecteurs interceptés, ajoutés à la fin de leurs interceptions
        self.V = None   # contient les vecteurs sélectionnés à l'itération présente

        self.model = None # contient un model qui fait la sélection des impulsions
        self.model_param = None
        if model_path is not None:
            self.load_model(model_path)

        self.T = 0
        self.running = True

    def SelectPulses(self, Input):
        if Input.shape[0] == 0:
            self.V = Input

        else:
            if self.model is None:

                # We make a mask, BM, that is 0 only if the pulse is selected
                Frequencies = torch.matmul(Input, self.weight_f)
                Levels = torch.matmul(Input, self.weight_l)

                BF = torch.abs(Frequencies.unsqueeze(-1) - Frequencies.unsqueeze(-2)) < self.sensitivity
                BN = (Levels.unsqueeze(-1) > Levels.unsqueeze(-2))
                BM = torch.sum(BF * BN, dim=-2) == 0

                # The selected pulses are then sorted by decreasing level
                Selected = Input[BM]
                Levels = torch.matmul(Selected, self.weight_l)
                Orders = Levels.argsort(dim=-1, descending=True)[:self.n_sat]
                self.V = Selected[Orders]

            else:
                len_in = self.model_param['len_in']

                inp = pad(Input, (0, 0, 0, len_in - len(Input))).unsqueeze(0)

                Output = self.model(inp)

                inp = torch.nn.functional.pad(inp, [0, 0, 0, 1])

                Diff = Output.unsqueeze(dim=1) - inp.unsqueeze(dim=2)
                Dist = torch.norm(Diff, dim=-1)
                Arg = (inp.shape[1] - 1) - torch.argmin(Dist.flip(dims=[1]), dim=1)
                Selected = []
                for arg in Arg[0]:
                    if sum(inp[0][arg]) != 0:
                        Selected.append(inp[0][arg].tolist())
                self.V = torch.tensor(Selected)

    def Process(self, Input):
        self.T += 1

        # on sélectionne les vecteurs contenus dans les informations interceptées à l'instant présent
        self.SelectPulses(torch.tensor(Input, dtype=torch.float))


        if self.V.shape[0] == 0:
            if self.P == []:
              self.running = False

            j = 0
            while j < len(self.TM):
                # le vecteur de suivi n° j n'est plus mise-à-jour : l'information interceptée est complète
                if self.T - self.TM[j] == 2:
                    TI, TM, P = self.TI.pop(j), self.TM.pop(j), self.P.pop(j)
                    # on ajoute deux informations : le temps d'intercepté et la date de début d'interception
                    P += [TM - TI, len(self.R) - TI]
                    # le vecteur intercepté est ajouté à self.R
                    self.R.append(P)
                else:
                    j += 1
            return

        # si aucun vecteur de suivi n'existe, chaque information interceptée est envoyée dans un nouveau vecteur de suivi
        if self.P == []:
            self.running = True
            for i in range(len(self.V)):
                self.P.append(self.V[i].tolist())
                self.TI.append(self.T-1)
                self.TM.append(self.T)
            return

        # sinon, on compare les informations interceptées aux vecteurs de suivi pour traitement
        self.running = True
        fV = torch.matmul(self.V, self.weight_f)
        lV = torch.matmul(self.V, self.weight_l)
        fP = torch.matmul(torch.tensor(self.P), self.weight_f)
        lP = torch.matmul(torch.tensor(self.P), self.weight_l)
        correlation = torch.abs(fV.unsqueeze(-1) - fP.unsqueeze(-2)) < self.sensitivity
        m = len(self.P)
        for i in lV.argsort(descending=True):
            selected_instance = []
            for j in range(m):
                if correlation[i, j] and (self.TM[j] < self.T):
                    selected_instance.append(j)
            # traitement 1 : pas de corrélation entre V[i] et P : création d'un nouveau vecteur de suivi
            if selected_instance == []:
                if len(self.P) < self.n_mes:
                    self.P.append(self.V[i].tolist())
                    self.TM.append(self.T)
                    self.TI.append(self.T - 1)
            # traitement 2 : corrélation entre V[i] et plusieurs vecteurs de suivi P[j1], P[j2] ...
            else:
                # on détermine le vecteur de suivi avec corrélation de niveau le plus haut
                Levels = lP[selected_instance]
                k = selected_instance[torch.argmax(Levels)]
                # on met à jour le vecteur de suivi P[k] correspondant
                self.TM[k] = self.T
                self.P[k] = (torch.tensor(self.P)[k] + (fV[i] - fP[k]) * self.weight_f).tolist()

        j = 0
        while j < len(self.TM):
            # le vecteur de suivi n° j n'est plus mise-à-jour : l'information interceptée est complète
            if self.T - self.TM[j] == 2:
                TI, TM, P = self.TI.pop(j), self.TM.pop(j), self.P.pop(j)
                # on ajoute deux informations : le temps d'interception et la date de début d'interception
                P += [TM - TI, len(self.R) - TI]
                # le vecteur intercepté est ajouté à self.R
                self.R.append(P)
            else:
                j += 1

    def load_model(self, model_path):
        import os
        from RankAI.V4.Vecteurs.Ranker import Network
        from Tools.XMLTools import loadXmlAsObj
        param = loadXmlAsObj(os.path.join(model_path, 'param'))

        model = Network(
            n_encoder=param['n_encoder'], len_in=param['len_in'], len_latent=param['len_out'], d_in=param['d_in'],
            d_att=param['d_att'], WidthsEmbedding=param['WidthsEmbedding'],
            n_heads=param['n_heads'], norm=param['norm'], dropout=param['dropout']
        )

        weight_l = torch.load(os.path.join(model_path, 'WeightN'))
        weight_f = torch.load(os.path.join(model_path, 'WeightF'))

        if (sum(weight_l - self.weight_l) != 0) or (sum(weight_f - self.weight_f) != 0):
            raise ValueError('weight of the simulation do not match weight using for training the model')

        model.load_state_dict(torch.load(os.path.join(model_path, 'Best_network')))
        self.model = model
        self.model_param = param