import numpy as np

class Platform():
    def __init__(self, StartingTime=0, EndingTime=0):
        self.Pulses = []
        self.VisiblePulses = []
        self.StartingTime = StartingTime
        self.EndingTime = EndingTime

    def AddPulse(self, Pulse):
        self.Pulses.append(Pulse)

    def DelPulse(self, Id):
        del(self.Pulses[Id])

    def DelVisiblePulse(self, Id):
        del(self.VisiblePulses[Id])

    def IsEmpty(self):
        return self.Pulses==[]

class Processor():
    def __init__(self, FreqThreshold=1.5, Fe=500, FreqSensibility=1, SaturationThreshold=5):
        self.FreqThreshold = FreqThreshold
        self.FreqSensibility = FreqSensibility
        self.Fe = Fe
        self.SaturationThreshold = SaturationThreshold

    # Cette fonction supprime les impulsions qui doivent l'être suite à leurs interaction sur un palier
    def Interaction(self, Platform):
        # Si le palier est vide, on n'a rien à faire
        if Platform.IsEmpty():
            Platform.VisiblePulses = []
            return None

        # Si l'impulsion de plus haut niveau dépasse le seuil de saturation, elle est la seule conservée
        IdLvlMax = np.argmax([Pulse.Level for Pulse in Platform.Pulses])
        if Platform.Pulses[IdLvlMax].Level > self.SaturationThreshold:
            Platform.VisiblePulses = [Platform.Pulses[IdLvlMax]]
            return None
        # Si deux impulsions ont des fréquences repliées proches, on supprime celle de plus bas niveau
        Platform.VisiblePulses = Platform.Pulses.copy()
        for Pulse in Platform.VisiblePulses:
            FreqRep = Pulse.GetFreq(Platform.StartingTime) % self.Fe
            for OtherPulse in Platform.VisiblePulses:
                if OtherPulse != Pulse:
                    FreqRepOther = OtherPulse.GetFreq(Platform.StartingTime) % self.Fe
                    if (abs(FreqRep-FreqRepOther) < self.FreqSensibility) and (Pulse.Level > OtherPulse.Level):
                        # print('\n')
                        # print(Pulse, ' is hiding ', OtherPulse)
                        # print('Folded Frequency of hiding pulse : {}'.format(FreqRep))
                        # print('Folded Frequency of hided pulse : {}'.format(FreqRepOther))
                        # print('\n')
                        Platform.DelVisiblePulse(Platform.VisiblePulses.index(OtherPulse))

    def Correlation(self, Platform, Trackers):
        FreqTrackers = np.array([tracker.FreqCur for tracker in Trackers])
        LevelTrackers = np.array([tracker.Level for tracker in Trackers])
        FreqPulses = np.array([pulse.GetFreq(Platform.StartingTime) for pulse in Platform.Pulses])
        Comp = np.abs(np.expand_dims(FreqTrackers, axis=0)-np.expand_dims(FreqPulses, axis=1)) < self.FreqThreshold
        Available = np.array([Tracker.IsTaken for Tracker in Trackers])
        Comp = Comp * Available


        Id = list(np.argmax(Comp*np.expand_dims(LevelTrackers, axis=0), axis=1))
        for i in range(len(Id)):
            if sum(Comp[i]) == 0:
                Id[i] = False
        return Id

    def RunPlatform(self, Platform, Trackers):
        # On crée la liste des impulsions visibles sur le palier
        self.Interaction(Platform)

        # On crée la liste de corrélation entre les mesureurs et TOUTES les impulsions du palier
        Id = self.Correlation(Platform, Trackers)
        for i in range(len(Platform.Pulses)):
            if Platform.Pulses[i] in Platform.VisiblePulses:
                # Si l'impulsion d'indice "i" est visible
                if Id[i] is False:
                    # Si l'impulsion d'indice "i" ne corrèle avec aucun mesureur
                    # On essaie d'ouvrir un mesureur
                    for Tracker in Trackers:
                        if Tracker.Open(Id=i):
                            # on arrête d'essayer d'ouvrir les mesureurs si on arrive à en ouvrir un
                            break

                else:
                    # Sinon on met à jour le mesureur (pas les fréquences min et max car le mesureur peut encore être coupé si il est trop long)
                    Trackers[Id[i]].Update(Id=i)
        for Tracker in Trackers:
            if Tracker.IsTaken:
                Tracker.Check(Platform)
