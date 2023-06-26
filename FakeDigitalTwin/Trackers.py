class Tracker():
    def __init__(self, Id, MaxAge, parent):
        self.id = Id
        self.MaxAge = MaxAge
        self.parent = parent
        self.Release(flag=None)

    def Open(self, Id, TOA=None, flag=None):
        self.Histogram = [Id]
        if self.IsTaken:
            return False
        self.IsTaken = True
        self.flag = flag
        self.TOA = self.parent.Platform.StartingTime
        if TOA is not None:
            self.TOA = TOA
        self.TOE = self.parent.Platform.EndingTime
        self.FreqMin = min(self.parent.Platform.Pulses[Id].GetFreq(self.TOE), self.parent.Platform.Pulses[Id].GetFreq(self.TOA))
        self.FreqCur = self.parent.Platform.Pulses[Id].GetFreq(self.TOE)
        self.FreqMax = max(self.parent.Platform.Pulses[Id].GetFreq(self.TOE), self.parent.Platform.Pulses[Id].GetFreq(self.TOA))
        self.Level = self.parent.Platform.Pulses[Id].Level
        return True

    def Release(self, flag=None):
        self.IsTaken = False
        return flag

    def Update(self, Id):
        self.Histogram.append(Id)
        platform = self.parent.Platform
        self.TOE = platform.EndingTime
        CurrentFreq = platform.Pulses[Id].GetFreq(self.TOE)
        self.FreqMin = min(self.FreqMin, CurrentFreq)
        self.FreqCur = CurrentFreq
        self.FreqMax = max(self.FreqMax, CurrentFreq)

    def CheckAge(self):
        Age = self.TOA - self.TOE
        if Age > self.MaxAge:
            self.TOE = self.TOA + self.MaxAge
            self.Release(flag='CW')
            self.Open(Id=self.Histogram[-1], TOA=self.TOE, flag='CW')

class Pulse():
    def __init__(self, FreqStart, FreqEnd, TOA, LI, Level):
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