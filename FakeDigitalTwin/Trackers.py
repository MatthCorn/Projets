class Tracker():
    def __init__(self, Id, parent):
        self.id = Id
        self.parent = parent
        self.IsTaken = False

    def Open(self, DNA, flag):
        self.IsTaken = True
        self.flag = flag
        self.TOA = self.parent.PlatformStartingTime
        self.TOE = self.parent.PlatformEndingTime
        self.Freq = DNA['FREQ']
        self.Level = DNA['LEVEL']

    def Release(self, flag):
        self.IsTaken = False
        return flag
