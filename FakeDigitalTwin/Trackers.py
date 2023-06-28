class Tracker():
    def __init__(self, Id, MaxAge, HoldingTime=3, parent=None):
        self.id = Id
        self.MaxAge = MaxAge
        self.HoldingTime = HoldingTime
        self.parent = parent
        self.IsTaken = False
        self.FreqCur = -float('inf')
        self.Level = 0
        self.TOA = 0
        self.TOE = -1
        self.flags = []
        self.Histogram = []

    def Open(self, Id, TOA=None, flag=[]):
        # L'implémentation de l'histogramme ne permet pas de savoir quelles impulsions sont présentes dans le mesureur
        # L'histogramme ne sert qu'à réouvrir correctement un mesureur en cas de CW (pour l'instant)
        self.Histogram = [Id]
        if self.IsTaken:
            return False
        self.IsTaken = True
        self.flags = set(flag)
        self.TOA = self.parent.Platform.StartingTime
        if TOA is not None:
            self.TOA = TOA
        if self.TOE == self.TOA:
            # Si le mesureur vient d'être libéré (donc il y avait saturation des mesureurs)
            # On ajoute le flag "troncature avant"
            self.flags.add('TroncAv')
        self.TOE = self.parent.Platform.EndingTime
        self.FreqMin = self.parent.Platform.VisiblePulses[Id].GetFreq(self.TOA)
        self.FreqCur = self.parent.Platform.VisiblePulses[Id].GetFreq(self.TOE)
        self.FreqMax = self.parent.Platform.VisiblePulses[Id].GetFreq(self.TOA)
        self.Level = self.parent.Platform.VisiblePulses[Id].Level
        return True

    def Release(self, flag=None):
        if flag is not None:
            self.flags.add(flag)
        self.parent.PDWs.append(self.Emission())
        self.IsTaken = False

    def Update(self, Id):
        self.Histogram.append(Id)
        platform = self.parent.Platform
        self.TOE = platform.EndingTime
        CurrentFreq = platform.VisiblePulses[Id].GetFreq(self.TOE)
        self.FreqCur = CurrentFreq

    def Check(self, platform):
        # Si la platform est vide, tous les mesureurs sont fermés
        if platform.IsEmpty():
            self.Release()
            return None

        # On vérifie que le mesureur n'est pas ouvert depuis trop longtemps
        Age = self.TOE - self.TOA
        if Age > self.MaxAge:
            self.TOE = self.TOA + self.MaxAge

            # On met à jour les fréquences min et max
            Id = self.Histogram[-1]
            CurrentFreq = platform.VisiblePulses[Id].GetFreq(self.TOE)
            self.FreqMin = min(self.FreqMin, CurrentFreq)
            self.FreqMax = max(self.FreqMax, CurrentFreq)

            self.Release(flag='CW')
            self.Open(Id=Id, TOA=self.TOE, flag=['CW'])
            self.Check(platform)
            return None

        # On vérifie que le mesureur n'a pas perdu la trace de son impulsion
        if platform.EndingTime - self.TOE > self.HoldingTime:
            self.Release()
            self.TOE = platform.EndingTime
            return None

        # Sinon on peut mettre à jour les fréquences min et max
        Id = self.Histogram[-1]
        CurrentFreq = platform.VisiblePulses[Id].GetFreq(self.TOE)
        self.FreqMin = min(self.FreqMin, CurrentFreq)
        self.FreqMax = max(self.FreqMax, CurrentFreq)
        return None

    def Emission(self):
        # return None
        PDW = {'TOA': self.TOA, 'LI': self.TOE - self.TOA, 'FreqMin': self.FreqMin, 'FreqMax': self.FreqMax, 'flags': self.flags}
        return PDW

    def __repr__(self):
        return '(Tracker : Taken={} ; TOA={} ; TOE={} ; Histogram={} ; Freq={})'.format(self.IsTaken, self.TOA, self.TOE, self.Histogram, self.FreqCur)
class Pulse():
    def __init__(self, FreqStart=0, FreqEnd=0, TOA=0, LI=0, Level=0):
        self.FreqStart = FreqStart
        self.FreqEnd = FreqEnd
        self.TOA = TOA
        self.LI = LI
        self.Level = Level

    def GetFreq(self, t):
        TOA, TOE = self.TOA, self.TOA+self.LI
        if (t < TOA) or (t > TOE):
            raise ValueError('"t" is not in the pulse emission time interval')
        return (self.FreqStart*(TOE-t) + self.FreqEnd*(t-TOA))/self.LI

    def __str__(self):
        return '(TOA={}, LI={}, FreqStart={}, FreqEnd={})'.format(self.TOA, self.LI, self.FreqStart, self.FreqEnd)

    def __repr__(self):
        return '(Pulse : TOA={} ; LI={} ; FreqStart={} ; FreqEnd={})'.format(self.TOA, self.LI, self.FreqStart, self.FreqEnd)
