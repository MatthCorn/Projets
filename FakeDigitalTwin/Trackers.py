import numpy as np

class TrackersList():
    def __init__(self, NbTrackers=5, parent=None):
        self.NbTrackers = NbTrackers
        self.Trackers = []
        self.parent = parent


    def SetParam(self, Fe=5000, FreqThreshold=10, MaxAgeTracker=10):
        self.Fe = Fe
        self.FreqThreshold = FreqThreshold
        self.MaxAgeTracker = MaxAgeTracker

    def CheckCompatibility(self, Pulse):
        FreqTrackers = np.array([tracker['FREQ'] for tracker in self.Trackers])
        LevelTrackers = np.array([tracker['LEVEL'] for tracker in self.Trackers])
        FreqPulse = Pulse['FREQ']
        Comp = np.abs(FreqTrackers-FreqPulse) > self.FreqThreshold
        if np.sum(Comp) == 0:
            return False
        Id = np.argmax(Comp*LevelTrackers)
        return Id

    def RunPlatform(self, Pulse):
        Id = self.CheckCompatibility(Pulse)
        if Id is False:
            if len(self.Trackers) < self.NbTrackers:
                self.OpenTracker(Pulse)
            else:
                self.UpdateTracker(Id, Pulse)
        self.CheckAgeTrackers()

class Tracker():
    def __init__(self, parent):
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
