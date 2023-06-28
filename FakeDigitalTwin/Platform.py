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

    def IsEmpty(self):
        return self.Pulses==[]

class Processor():
    def __init__(self, FreqThreshold=1.5, Fe=500, MaxAgeTracker=10, FreqSensibility=1):
        self.FreqThreshold = FreqThreshold
        self.FreqSensibility = FreqSensibility
        self.Fe = Fe
        self.MaxAgeTracker = MaxAgeTracker

    # Cette fonction supprime les impulsions qui doivent l'être suite à leurs interaction sur un palier
    def Interaction(self, Platform):
        # Si deux impulsions ont des fréquences repliées proches, on supprime celle de plus bas niveau
        Platform.VisiblePulses = Platform.Pulses.copy()
        return None
        for Pulse in Platform.VisiblePulses:
            FreqRep = Pulse.GetFreq(Platform.StartingTime) % self.Fe
            for OtherPulse in Platform.VisiblePulses:
                if OtherPulse != Pulse:
                    FreqRepOther = OtherPulse.GetFreq(Platform.StartingTime) % self.Fe
                    if (abs(FreqRep-FreqRepOther) < self.FreqSensibility) and (Pulse.Level > OtherPulse.Level):
                        Platform.DelPulse(Platform.VisiblePulses.index(OtherPulse))

    def Correlation(self, Platform, Trackers):
        FreqTrackers = np.array([tracker.FreqCur for tracker in Trackers])
        LevelTrackers = np.array([tracker.Level for tracker in Trackers])
        P = Platform.VisiblePulses
        ST = Platform.StartingTime
        ET = Platform.EndingTime
        FreqPulses = np.array([pulse.GetFreq(Platform.StartingTime) for pulse in P])
        Comp = np.abs(np.expand_dims(FreqTrackers, axis=0)-np.expand_dims(FreqPulses, axis=1)) < self.FreqThreshold
        Available = np.array([Tracker.IsTaken for Tracker in Trackers])
        Comp = Comp * Available


        Id = list(np.argmax(Comp*np.expand_dims(LevelTrackers, axis=0), axis=1))
        for i in range(len(Id)):
            if sum(Comp[i]) == 0:
                Id[i] = False
        return Id

    def RunPlatform(self, Platform, Trackers):
        self.Interaction(Platform)
        Id = self.Correlation(Platform, Trackers)
        for i in range(len(Platform.VisiblePulses)):
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
