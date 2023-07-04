from FakeDigitalTwin.Trackers import Pulse
from FakeDigitalTwin.Simulator import DigitalTwin
import numpy as np

size = 100000

# On se donne un scénario de 1000 unités de temps
TOA = 1000 * np.random.random(size=size)

####################################################################################################################
# Le temps de maintien max est de 2 unités de temps, on veut que 90% des impulsions soit moins longues

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

AntP = [Pulse(TOA=TOA[k], LI=LI[k], Level=Level[k], FreqStart=FreqStart[k], FreqEnd=FreqEnd[k]) for k in range(size)]

# Nombre de mesureurs
NbMaxTrackers = 6
# Seuil de différence de fréquences repliées en dessous duquel deux impulsions se gênent
FreqSensibility = 1
# Seuil de différence de fréquence entre un mesureur et une impulsion en dessous duquel ces derniers sont associés
FreqThreshold = 1.5
# Fréquence d'échantillionage du système
Fe = 3
# Temps de maintien d'un mesureur
MaxAgeTracker = 10
# Seuil du niveau au dessus duquel les impulsions saturent le palier
SaturationThreshold = 10


DT = DigitalTwin(NbMaxTrackers=NbMaxTrackers, FreqThreshold=FreqThreshold, Fe=Fe, MaxAgeTracker=MaxAgeTracker, FreqSensibility=FreqSensibility, SaturationThreshold=SaturationThreshold)
DT.forward(AntP)