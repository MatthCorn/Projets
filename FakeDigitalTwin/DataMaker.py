from Tools.XMLTools import saveObjAsXml
from FakeDigitalTwin.Simulator import DigitalTwin
from FakeDigitalTwin.Pulse import Pulse
import numpy as np
import os
from tqdm import tqdm


Param = {
    'Fe_List': [5.1, 5, 4.9, 4.8],
    'Duree_max_impulsion': 4,
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
    'N_DetEl': 12,
    'Seuil_ecart_freq': 5e-3,
    'Duree_maintien_max': 0.2,
    'N_mesureurs_max': 8,
    'PDW_tries': False,
}

def MakeData(Batch_size, density, seed, name, param=Param, return_data=False):
    BatchPulses = []
    BatchPDWs = []

    if seed is not None:
        np.random.seed(seed)

    for _ in tqdm(range(Batch_size)):
        TMax = 100
        size = int(TMax * density / 1.06)

        # On se donne un scénario de TMax unités de temps
        # On a donc en moyenne "density" impulsions en même temps
        TOA = TMax * np.sort(np.random.random(size=size))

        ####################################################################################################################
        # Pour un radar pulsé, si l'unité de temps est la microseconde, la durée de l'impulsion est entre 0.1 et 1
        # On ajoute aussi 3% d'impulsions en onde continue, de LI comprise entre 15 et 20
        ####################################################################################################################
        LIcourte = np.random.uniform(0.1, 1, int(size * 0.97))
        LILongue = np.random.uniform(15, 20, size - int(size * 0.97))
        LI = np.concatenate([LILongue, LIcourte])
        np.random.shuffle(LI)

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
        Level = np.random.gamma(shape=2, scale=1.725, size=size)

        # Les fréquences se trouvent entre 0 et 10 et peuvent varier le long d'une impulsion de 0.05 unité de fréquence
        dF = 0.05 * (2 * np.random.random(size=size) - 1)
        FreqMoy = 9 * np.random.random(size=size) + 0.5
        FreqStart = FreqMoy + dF
        FreqEnd = FreqMoy - dF

        AntP = [Pulse(TOA=round(TOA[k], 3), LI=round(LI[k], 3), Level=round(Level[k], 3), FreqStart=round(FreqStart[k], 3),
                      FreqEnd=round(FreqEnd[k], 3), Id=k) for k in range(size)]

        DT = DigitalTwin(Param=param)

        DT.forward(AntP)

        BatchPulses.append([{'TOA': Pulse.TOA, 'LI': Pulse.LI, 'Level': Pulse.Level, 'FreqStart': Pulse.FreqStart,
                       'FreqEnd': Pulse.FreqEnd, 'Id': Pulse.Id} for Pulse in AntP])

        BatchPDWs.append(DT.PDWs)

    if return_data:
        return BatchPulses, BatchPDWs

    cwd = os.getcwd()
    (path, dir) = os.path.split(cwd)
    while dir != 'Projets':
        (path, dir) = os.path.split(path)
    cwd = os.path.join(path, dir, 'FakeDigitalTwin')

    saveObjAsXml(BatchPulses, os.path.join(cwd, 'Data', name+'PulsesAnt.xml'))
    saveObjAsXml(BatchPDWs, os.path.join(cwd, 'Data', name+'PDWsDCI.xml'))


def MakeDataHI(Batch_size, density, seed, name, param=Param, return_data=False):
    BatchPulses = []
    BatchPDWs = []

    if seed is not None:
        np.random.seed(seed)

    TMax = 100
    size = int(TMax * density / 1.06)

    for _ in tqdm(range(Batch_size)):

        TOA = TMax * np.sort(np.random.random(size=size))

        LIcourte = np.random.uniform(0.1, 1, int(size * 0.97))
        LILongue = np.random.uniform(15, 20, size - int(size * 0.97))
        LI = np.concatenate([LILongue, LIcourte])
        np.random.shuffle(LI)

        Level = np.random.gamma(shape=2, scale=1.725, size=size)

        Fe = Param['Fe_List'][0]
        FreqRepMoy = Fe/2 * np.random.rand()
        FreqRep = (0.4 * np.random.random(size=size) - 0.2 + FreqRepMoy) % Fe/2

        FreqMax, FreqVar = 10, 0.1
        FreqMoy = []
        for freq in FreqRep:
            FreqPossible = [-freq + k*Fe for k in range(1, int(FreqMax//Fe) + 2)] + [freq + k*Fe for k in range(int(FreqMax//Fe) + 1)]
            FreqPossible = [freq for freq in FreqPossible if FreqVar < freq < (FreqMax - FreqVar)]
            FreqMoy.append(FreqPossible[np.random.randint(len(FreqPossible))])
        FreqMoy = np.array(FreqMoy)
        dF = FreqVar * (2 * np.random.random(size=size) - 1)
        FreqStart = FreqMoy + dF
        FreqEnd = FreqMoy - dF

        AntP = [Pulse(TOA=round(TOA[k], 3), LI=round(LI[k], 3), Level=round(Level[k], 3), FreqStart=round(FreqStart[k], 3), FreqEnd=round(FreqEnd[k], 3), Id=k) for k in range(size)]

        DT = DigitalTwin(Param=param)

        DT.forward(AntP)

        BatchPulses.append([{'TOA': Pulse.TOA, 'LI': Pulse.LI, 'Level': Pulse.Level, 'FreqStart': Pulse.FreqStart,
                       'FreqEnd': Pulse.FreqEnd, 'Id': Pulse.Id} for Pulse in AntP])

        BatchPDWs.append(DT.PDWs)

    if return_data:
        return BatchPulses, BatchPDWs

    cwd = os.getcwd()
    (path, dir) = os.path.split(cwd)
    while dir != 'Projets':
        (path, dir) = os.path.split(path)
    cwd = os.path.join(path, dir, 'FakeDigitalTwin')

    saveObjAsXml(BatchPulses, os.path.join(cwd, 'Data', name+'PulsesAnt.xml'))
    saveObjAsXml(BatchPDWs, os.path.join(cwd, 'Data', name+'PDWsDCI.xml'))

