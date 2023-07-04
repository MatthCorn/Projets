from FakeDigitalTwin.Trackers import Pulse
from FakeDigitalTwin.Simulator import DigitalTwin
import numpy as np
import sys

sys.setrecursionlimit(2147483647)

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

# On essaie d'avoir une distribution qui occasionne un bug assez tôt
Bool = False
while Bool:

    size = 200

    # On se donne un scénario de 200 unités de temps
    TOA = 100 * np.sort(np.random.random(size=size))

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


    try:
        DT = DigitalTwin(NbMaxTrackers=NbMaxTrackers, FreqThreshold=FreqThreshold, Fe=Fe, MaxAgeTracker=MaxAgeTracker,
                         FreqSensibility=FreqSensibility, SaturationThreshold=SaturationThreshold)
        DT.forward(AntP)
    except:
        Bool = DT.TimeId > 10

    if not Bool:
        AntP = AntP[:10]
        print(AntP)

# Voici un scénario qui bloque à la 8ème impulsion (pour étudier le fonctionnement pas à pas)
AntP = [Pulse(TOA=1.2524689885551665, LI=0.2832551730857815, Level=8.113107971284112, FreqStart=6.564792031326859, FreqEnd=7.059853818612322),
        Pulse(TOA=1.5950971264305114, LI=2.22421091221156, Level=9.390587242445003, FreqStart=0.7169653815803326, FreqEnd=0.6984591358975528),
        Pulse(TOA=1.9171509282674304, LI=0.606617418036221, Level=7.2535746407672, FreqStart=3.509338603326864, FreqEnd=3.2588071147149686),
        Pulse(TOA=3.8018973512388032, LI=1.269383481428134, Level=2.807553958555972, FreqStart=5.655526751263485, FreqEnd=4.720742218662277),
        Pulse(TOA=4.828613542102889, LI=1.6315610980714874, Level=5.273735340444493, FreqStart=0.950425271971837, FreqEnd=1.7026606373576731),
        Pulse(TOA=5.814432731560659, LI=2.5578484830683323, Level=4.781261729101505, FreqStart=8.386752994871255, FreqEnd=8.583737137868752),
        Pulse(TOA=5.836869695362179, LI=0.47760313073885463, Level=3.024411591567953, FreqStart=8.63896025618675, FreqEnd=7.876013389905055),
        Pulse(TOA=6.177568458299354, LI=0.9256336982340553, Level=12.558812898448158, FreqStart=8.47515745407202, FreqEnd=7.865027404580393),
        Pulse(TOA=8.262818432965446, LI=1.9376003878413062, Level=2.847191170471996, FreqStart=2.260764434946223, FreqEnd=1.970447731402996),
        Pulse(TOA=8.65722462619537, LI=0.9580911504655671, Level=6.347866335115073, FreqStart=9.496471722571432, FreqEnd=9.10210445115089)]

DT = DigitalTwin(NbMaxTrackers=NbMaxTrackers, FreqThreshold=FreqThreshold, Fe=Fe, MaxAgeTracker=MaxAgeTracker,
                         FreqSensibility=FreqSensibility, SaturationThreshold=SaturationThreshold)

DT.forward(AntP)
