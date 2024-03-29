from FakeDigitalTwin.SimulatorIter import DigitalTwin
from Tools.XMLTools import saveObjAsXml
from FakeDigitalTwin.Trackers import Pulse
import os
import numpy as np
from tqdm import tqdm


def MakeData(Batch_size, density, seed, name, return_data=False):
    BatchPulses = []
    BatchPDWs = []

    if seed is not None:
        np.random.seed(seed)

    for iter in tqdm(range(Batch_size)):
        # Nombre de mesureurs
        NbMaxTrackers = 4
        # Seuil de différence de fréquences repliées en dessous duquel deux impulsions se gênent
        FreqSensibility = 0.1
        # Seuil de différence de fréquence entre un mesureur et une impulsion en dessous duquel ces derniers sont associés
        FreqThreshold = 0.2
        # Fréquence d'échantillionage du système
        Fe = 3
        # Temps de maintien d'un mesureur (fermé automatiquement dès que plus vieux)
        MaxAgeTracker = 2
        # Seuil du niveau au dessus duquel les impulsions saturent le palier
        SaturationThreshold = 10
        # Temps de maintien max d'un mesureur sans voir son impulsion
        HoldingTime = 0.5

        size = int(100*density/(2*0.36))

        # On se donne un scénario de 100 unités de temps
        # On a donc en moyenne "density" impulsions en même temps
        TOA = 100 * np.sort(np.random.random(size=size))

        ####################################################################################################################
        # Le temps de maintien max d'un mesureur est de 2 unités de temps, on veut que 97.5% des impulsions soit moins longues

        # import numpy as np
        # t_max = 2
        # scale = 1
        # frac = 1
        # while frac > 0.1:
        #     scale *= 0.999
        #     np_gamma = np.random.gamma(shape=2, scale=scale, size=100000)
        #     frac = sum(np_gamma > t_max)/100000
        #     print(scale)
        #     print(frac)
        #     print('\n')
        #
        # scale = 0.36
        ####################################################################################################################
        LI = np.random.gamma(shape=2, scale=0.36, size=size) + 1e-2

        ####################################################################################################################
        # Le niveau de saturation est de 10 unités, on veut que 97.5% des impulsions soit moins fortes

        # import numpy as np
        # Lvl_max = 10
        # scale = 3
        # frac = 1
        # while frac > 0.1:
        #     scale *= 0.999
        #     np_gamma = np.random.gamma(shape=2, scale=scale, size=100000)
        #     frac = sum(np_gamma > Lvl_max)/100000
        #     print(scale)
        #     print(frac)
        #     print('\n')
        #
        # scale = 1.8
        ####################################################################################################################
        Level = np.random.gamma(shape=2, scale=1.8, size=size)

        # Les fréquences se trouvent entre 0 et 10 et peuvent varier le long d'une impulsion de 0.05 unité de fréquence
        dF = 0.05 * (2 * np.random.random(size=size) - 1)
        FreqMoy = 9 * np.random.random(size=size) + 0.5
        FreqStart = FreqMoy + dF
        FreqEnd = FreqMoy - dF

        AntP = [Pulse(TOA=round(TOA[k], 3), LI=round(LI[k], 3), Level=round(Level[k], 3), FreqStart=round(FreqStart[k], 3), FreqEnd=round(FreqEnd[k], 3), Id=k) for k in range(size)]


        DT = DigitalTwin(NbMaxTrackers=NbMaxTrackers, FreqThreshold=FreqThreshold, Fe=Fe, MaxAgeTracker=MaxAgeTracker,
                         FreqSensibility=FreqSensibility, SaturationThreshold=SaturationThreshold, HoldingTime=HoldingTime)

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


def MakeDataHI(Batch_size, density, seed, name, return_data=False):
    BatchPulses = []
    BatchPDWs = []

    if seed is not None:
        np.random.seed(seed)

    # Nombre de mesureurs
    NbMaxTrackers = 4
    # Seuil de différence de fréquences repliées en dessous duquel deux impulsions se gênent
    FreqSensibility = 0.1
    # Seuil de différence de fréquence entre un mesureur et une impulsion en dessous duquel ces derniers sont associés
    FreqThreshold = 0.2
    # Fréquence d'échantillionage du système
    Fe = 3
    # Temps de maintien d'un mesureur (fermé automatiquement dès que plus vieux)
    MaxAgeTracker = 2
    # Seuil du niveau au dessus duquel les impulsions saturent le palier
    SaturationThreshold = 10
    # Temps de maintien max d'un mesureur sans voir son impulsion
    HoldingTime = 0.5


    LI_max = 3 * MaxAgeTracker
    size = int(100*density/(LI_max/2))

    for iter in tqdm(range(Batch_size)):

        TOA = 100 * np.sort(np.random.random(size=size))
        LI = LI_max * np.random.random(size=size) + 1e-2
        Level = 0.8 * SaturationThreshold * np.random.random(size=size) + 0.5 * SaturationThreshold


        FreqRepMoy = Fe * np.random.rand()
        FreqRep = (0.4 * np.random.random(size=size) - 0.2 + FreqRepMoy) % Fe

        FreqMax, FreqVar = 10, 0.1
        FreqMoy = []
        for freq in FreqRep:
            FreqPossible = [-freq + k*Fe for k in range(1, FreqMax//Fe + 2)] + [freq + k*Fe for k in range(FreqMax//Fe + 1)]
            FreqPossible = [freq for freq in FreqPossible if FreqVar < freq < (FreqMax - FreqVar)]
            FreqMoy.append(FreqPossible[np.random.randint(len(FreqPossible))])
        FreqMoy = np.array(FreqMoy)
        dF = FreqVar * (2 * np.random.random(size=size) - 1)
        FreqStart = FreqMoy + dF
        FreqEnd = FreqMoy - dF

        AntP = [Pulse(TOA=round(TOA[k], 3), LI=round(LI[k], 3), Level=round(Level[k], 3), FreqStart=round(FreqStart[k], 3), FreqEnd=round(FreqEnd[k], 3), Id=k) for k in range(size)]


        DT = DigitalTwin(NbMaxTrackers=NbMaxTrackers, FreqThreshold=FreqThreshold, Fe=Fe, MaxAgeTracker=MaxAgeTracker,
                         FreqSensibility=FreqSensibility, SaturationThreshold=SaturationThreshold, HoldingTime=HoldingTime)

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

