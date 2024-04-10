class Pulse():
    def __init__(self, FreqStart=0, FreqEnd=0, TOA=0, LI=0, Level=0, Id=0):
        self.FreqStart = FreqStart
        self.FreqEnd = FreqEnd
        self.TOA = TOA
        self.LI = LI
        self.Level = Level
        self.Id = Id

    def GetFreq(self, t):
        TOA, TOD = self.TOA, self.TOA+self.LI
        if (t < TOA) or (t > TOD):
            raise ValueError('"t" is not in the pulse emission time interval')
        return (self.FreqStart*(TOD-t) + self.FreqEnd*(t-TOA))/self.LI

    def __str__(self):
        return 'Pulse(TOA={}, LI={}, Level={}, FreqStart={}, FreqEnd={}, Id={})'.format(self.TOA, self.LI, self.Level, self.FreqStart, self.FreqEnd, self.Id)

    def __repr__(self):
        return 'Pulse(TOA={}, LI={}, Level={}, FreqStart={}, FreqEnd={}, Id={})'.format(self.TOA, self.LI, self.Level, self.FreqStart, self.FreqEnd, self.Id)