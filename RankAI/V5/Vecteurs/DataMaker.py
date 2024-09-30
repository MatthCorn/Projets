class Pulse(list):
    def __init__(self, li):
        super().__init__(li)
    def GetFreq(self, t):
        return self.__getitem__(2)
    def GetLevel(self):
        return self.__getitem__(3)

    @property
    def Level(self):
        return self.__getitem__(3)

class Plateau():
    def __init__(self, li=[]):
        self.Pulses = li
        self.LevelPulses = self.Pulses.copy()
        self.LevelPulses.sort(key=Pulse.GetLevel)
        self.StartingTime = 0

    def IsEmpty(self):
        return self.Pulses == []

class BullshitParent():
    def __init__(self, Param):
        self.Param = Param

Param = {
    'Fe_List': [5.1, 5, 4.9, 4.8],
    'FMax': 2.5,
    'Duree_max_impulsion': 1,
    'Seuil_mono': 10,
    'Seuil_harmo': 8,
    'Seuil_IM': 8,
    'Seuil_sensi_traitement': 6,
    'Seuil_sensi': 1,
    'Contraste_geneur': 0.2,
    'Nint': 500,
    'Contraste_geneur_2': 1,
    'M1_aveugle': 2,
    'M2_aveugle': 2,
    'M_local': 2,
    'N_DetEl': 12,
    'Seuil_ecart_freq': 5e-3,
    'Duree_maintien_max': 0.2,
    'N_mesureurs_max': 8,
    'PDW_tries': False,
}

Param = BullshitParent(Param)

from FakeDigitalTwin.Processor import Processor

Proc = Processor(Param)

import numpy as np
from tqdm import tqdm
seed = 0
batch_size = 20000
NInput = 10
NOutput = 10
density = 5
Input = []
Output = []

if seed is not None:
    np.random.seed(seed)
for _ in tqdm(range(batch_size)):

    TMax = 10

    # On se donne un scénario de TMax unités de temps
    # On a donc en moyenne "density" impulsions en même temps
    TOA = TMax * np.sort(np.random.random(size=NInput))

    LI = np.random.uniform(0.1, 1, NInput)

    ####################################################################################################################
    # Le niveau de saturation est de 10 unités, on veut que 98% des impulsions soit moins fortes

    # import numpy as np
    # Lvl_max = 10
    # scale = 3
    # frac = 1
    # while frac > 0.02:
    #     scale *= 0.999
    #     np_gamma = np.random.gamma(shape=2, scale=scale, size=100000)
    #     frac = sum(np_gamma > Lvl_max)/100000
    #     print(scale)
    #     print(frac)
    #     print('\n')
    #
    # scale = 1.725
    ####################################################################################################################
    Level = np.random.gamma(shape=2, scale=1.725, size=NInput)

    Fe = Param.Param['Fe_List'][0]
    FreqMax = Param.Param['FMax']
    FreqRepMoy = min(Fe/2, FreqMax) * np.random.rand()
    FreqRep = (0.4 * np.random.random(size=NInput) - 0.2 + FreqRepMoy) % min(int(Fe/2), FreqMax)

    Freq = []

    for freq in FreqRep:
        FreqPossible = [-freq + k * Fe for k in range(1, int(FreqMax // Fe) + 2)] + \
                       [freq + k * Fe for k in range(int(FreqMax // Fe) + 1)]
        FreqPossible = [freq for freq in FreqPossible if 0 < freq < FreqMax]
        Freq.append(FreqPossible[np.random.randint(len(FreqPossible))])

    # Freq = 10 * np.random.random(size=NVec)

    li = [Pulse((round(TOA[k], 3), round(LI[k], 3), round(Freq[k], 3), round(Level[k], 3), )) for k in range(NInput)]

    Palier = Plateau(li)
    Input.append(Palier.Pulses)
    O = list(Proc.FreqAmbRemoval(Palier))
    O.sort(key=lambda x: x[3])
    while len(O) < NOutput:
        O.append([0]*4)
    Output.append(np.array(O))

Input = np.array(Input)
Output = np.array(Output)

print(sum([len(el) for el in Output])/len(Output))
print(Input[:2])
print(Output[:2])