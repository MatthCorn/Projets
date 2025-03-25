import torch

class Simulator:
    def __init__(self, dim, sensitivity, WeightF=None, WeightL=None):
        self.dim = dim
        self.sensitivity = sensitivity
        self.weight_f = WeightF if WeightF is not None else torch.tensor([1., 0.] + [0.] * (self.dim - 2))
        self.weight_f = self.weight_f / torch.norm(self.weight_f)
        self.weight_l = WeightL if WeightL is not None else torch.tensor([0., 1.] + [0.] * (self.dim - 2))
        self.weight_l = self.weight_l / torch.norm(self.weight_l)
        self.P = []     # contient les vecteurs de suivis, qui enregistrent itérativement les informations des vecteurs présents
        self.TM = []    # contient le temps de dernière mise-à-jour de chaque vecteur de suivi
        self.TI = []    # contient le temps d'apparition de chaque vecteur de suivi
        self.R = []     # contient tous les vecteurs interceptés, ajoutés à la fin de leurs interceptions
        self.V = None   # contient les vecteurs sélectionnés à l'itération présente
        self.T = 0
        self.running = True

    def SelectPulses(self, Input):
        if Input.shape[0] == 0:
            self.V = Input

        else:
            # We make a mask, BM, that is 0 only if the pulse is selected
            Frequencies = torch.matmul(Input, torch.tensor(self.weight_f))
            Levels = torch.matmul(Input, torch.tensor(self.weight_l))

            BF = torch.abs(Frequencies.unsqueeze(-1) - Frequencies.unsqueeze(-2)) < self.sensitivity
            BN = (Levels.unsqueeze(-1) > Levels.unsqueeze(-2))
            BM = torch.sum(BF * BN, dim=-2) == 0

            # The selected pulses are then sorted by decreasing level
            Selected = Input[BM]
            Levels = torch.matmul(Selected, torch.tensor(self.weight_l))
            Orders = Levels.argsort(dim=-1, descending=True)
            self.V = Selected[Orders]

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
        fP = torch.matmul(torch.tensor(self.P), self.weight_f)
        lP = torch.matmul(torch.tensor(self.P), self.weight_l)
        correlation = torch.abs(fV.unsqueeze(-1) - fP.unsqueeze(-2)) < self.sensitivity
        m = len(self.P)
        for i in range(len(self.V)):
            selected_instance = []
            for j in range(m):
                if correlation[i, j] and (self.TM[j] < self.T):
                    selected_instance.append(j)
            # traitement 1 : pas de corrélation entre V[i] et P : création d'un nouveau vecteur de suivi
            if selected_instance == []:
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
                # on ajoute deux informations : le temps d'intercepté et la date de début d'interception
                P += [TM - TI, len(self.R) - TI]
                # le vecteur intercepté est ajouté à self.R
                self.R.append(P)
            else:
                j += 1