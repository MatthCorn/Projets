import torch

class Simulator:
    def __init__(self, dim, sensitivity_fourier, sensitivity_sensor, WeightF=None, WeightL=None):
        self.dim = dim
        self.sensitivity_fourier = sensitivity_fourier
        self.sensitivity_sensor = sensitivity_sensor
        self.weight_f = WeightF if WeightF is not None else [1., 0.] + [0.] * (self.dim - 2)
        self.weight_l = WeightL if WeightL is not None else [0., 1.] + [0.] * (self.dim - 2)
        self.P = []
        self.TM = []
        self.TI = []
        self.R = []
        self.T = 0

    def SelectPulses(self, Input):
        # We make a mask, BM, that is 0 only if the pulse is selected
        Frequencies = torch.matmul(Input, torch.tensor(self.weight_f))
        Levels = torch.matmul(Input, torch.tensor(self.weight_l))

        BF = torch.abs(Frequencies.unsqueeze(-1) - Frequencies.unsqueeze(-2)) < self.sensitivity_fourier
        BN = (Levels.unsqueeze(-1) > Levels.unsqueeze(-2))
        BM = torch.sum(BF * BN, dim=-2) == 0

        # The selected pulses are then sorted by decreasing level
        Selected = Input[BM]
        Levels = torch.matmul(Selected, torch.tensor(self.weight_l))
        Orders = Levels.argsort(dim=-1, descending=True)
        Selected = Selected[Orders]

        return Selected

    def Process(self, Input):
        self.T += 1

        if Input == []:
            j = 0
            while j < len(self.TM):
                if self.T - self.TM[j] == 2:
                    TI, TM, P = self.TI.pop(j), self.TM.pop(j), self.P.pop(j)
                    P += [TM - TI, len(self.R) - TI]
                    self.R.append(P)
                else:
                    j += 1
            return

        V = self.SelectPulses(torch.tensor(Input, dtype=torch.float))

        if self.P == []:
            for i in range(len(V)):
                self.P.append(V[i].tolist())
                self.TI.append(self.T-1)
                self.TM.append(self.T)
            return

        fV = torch.matmul(V, torch.tensor(self.weight_f))
        fP = torch.matmul(torch.tensor(self.P), torch.tensor(self.weight_f))
        lP = torch.matmul(torch.tensor(self.P), torch.tensor(self.weight_l))
        correlation = torch.abs(fV.unsqueeze(-1) - fP.unsqueeze(-2)) < self.sensitivity_sensor
        m = len(self.P)
        for i in range(len(V)):
            selected_instance = []
            for j in range(m):
                if correlation[i, j] and (self.TM[j] < self.T):
                    selected_instance.append(j)
            if selected_instance == []:
                self.P.append(V[i].tolist())
                self.TM.append(self.T)
                self.TI.append(self.T - 1)
            else:
                Levels = lP[selected_instance]
                j = torch.argmax(Levels)
                self.TM[selected_instance[j]] = self.T

        j = 0
        while j < len(self.TM):
            if self.T - self.TM[j] == 2:
                TI, TM, P = self.TI.pop(j), self.TM.pop(j), self.P.pop(j)
                P += [TM - TI, len(self.R) - TI]
                self.R.append(P)
            else:
                j += 1
