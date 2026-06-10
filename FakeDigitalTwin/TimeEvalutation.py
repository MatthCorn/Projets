from FakeDigitalTwin.Simulator import DigitalTwin
from FakeDigitalTwin.Pulse import Pulse
import numpy as np
from time import time
from tqdm import tqdm

# Supposons que Pulse et DigitalTwin sont déjà importés/définis
N = 100  # nombre total d'impulsions du scénario
D = 3.  # nombre d'impulsions par ms pour un radar civil
N_monte = 5  # nombre de répétitions pour le calcul du temps d'inférence
LI_moy = 1e-3  # durée moyenne d'une impulsion en ms
expansion_factor = []  # liste des temps de simu
n_palier = []  # liste du nombre d'impulsions par palier (moyenne)

for k in tqdm(range(10, 2000)):

    # Création du scénario
    D_local = D * k  # nombre d'impulsions par ms pour le scénario
    T = N / D_local  # durée du scénario
    LI = [LI_moy + 1e-4 * np.random.random() for _ in range(N)]  # liste des durées d'impulsions
    F = [10 + 1 * np.random.random() for _ in range(N)]  # fréquence moyenne d'une impulsion : 10
    TOA = [T * np.random.random() for _ in range(N)]
    TOA.sort()
    AntP = [Pulse(TOA=TOA[i], LI=LI[i], FreqStart=F[i], FreqEnd=F[i], Level=5.5 * np.random.random() + 0.5) for i in range(N)]

    n_palier.append(LI_moy * D_local)
    Param = {
        'Fe_List': [5.1, 5, 4.9, 4.8],
        'Duree_max_impulsion': 5,
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
        'M_local': 5,
        'N_DetEl': 5,
        'Seuil_ecart_freq': 2e-2,
        'Duree_maintien_max': 1e-4,
        'N_mesureurs_max': 4,
        'PDW_tries': True,
    }

    t = 0
    for _ in range(N_monte):
        DT = DigitalTwin(Param)
        t = time() - t
        DT.forward(AntPulses=AntP)
        t = time() - t
    expansion_factor.append(t * 1000 / (N_monte * T))

import matplotlib.pyplot as plt
plt.plot(expansion_factor)
plt.show()