from FakeDigitalTwin.Trackers import Pulse
from FakeDigitalTwin.SimulatorIter import DigitalTwin
import numpy as np

# Nombre de mesureurs
NbMaxTrackers = 8
# Seuil de différence de fréquences repliées en dessous duquel deux impulsions se gênent
FreqSensibility = 0.1
# Seuil de différence de fréquence entre un mesureur et une impulsion en dessous duquel ces derniers sont associés
FreqThreshold = 0.3
# Fréquence d'échantillionage du système
Fe = 3
# Temps de maintien d'un mesureur
MaxAgeTracker = 2
# Seuil du niveau au dessus duquel les impulsions saturent le palier
SaturationThreshold = 10
# Temps de maintien max d'un mesureur
HoldingTime = 1

size = 20

# On se donne un scénario de 10 unités de temps
# On a donc en moyenne 4 impulsions en même temps
TOA = 10 * np.sort(np.random.random(size=size))

####################################################################################################################
# Le temps de maintien max d'un mesureur est de 2 unités de temps, on veut que 90% des impulsions soit moins longues

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
# scale = 0.5146
####################################################################################################################
LI = np.random.gamma(shape=2, scale=0.5146, size=size)

####################################################################################################################
# Le niveau de saturation est de 10 unités, on veut que 90% des impulsions soit moins fortes

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
# scale = 2.5716
####################################################################################################################
Level = np.random.gamma(shape=2, scale=2.5716, size=size)

# Les fréquences se trouvent entre 0 et 10 et peuvent varier le long d'une impulsion de 0.5 unité de fréquence
dF = 0.5 * (2 * np.random.random(size=size) - 1)
FreqMoy = 9 * np.random.random(size=size) + 0.5
FreqStart = FreqMoy + dF
FreqEnd = FreqMoy - dF

AntP = [Pulse(TOA=round(TOA[k], 3), LI=round(LI[k], 3), Level=round(Level[k], 3), FreqStart=round(FreqStart[k], 3), FreqEnd=round(FreqEnd[k], 3), Id=k) for k in range(size)]


DT = DigitalTwin(NbMaxTrackers=NbMaxTrackers, FreqThreshold=FreqThreshold, Fe=Fe, MaxAgeTracker=MaxAgeTracker,
                         FreqSensibility=FreqSensibility, SaturationThreshold=SaturationThreshold, HoldingTime=HoldingTime)

DT.forward(AntP)
print('nombre de PDWs :',len(DT.PDWs))