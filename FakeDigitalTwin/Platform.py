import numpy as np

class Platform():
    def __init__(self, Pulses, StartingTime, EndingTime):
        self.Pulses = Pulses
        self.StartingTime = StartingTime
        self.EndingTime = EndingTime

class Processor():
    def __init__(self, FreqThreshold=10, Fe=500, MaxAgeTracker=10):
        self.FreqThreshold = FreqThreshold
        self.Fe = Fe
        self.MaxAgeTracker = MaxAgeTracker

    def CheckCompatibility(self, Platform, Trackers):
        FreqTrackers = np.array([tracker['FREQ'] for tracker in Trackers])
        LevelTrackers = np.array([tracker['LEVEL'] for tracker in Trackers])
        FreqPulses = np.array([pulse['FREQ'] for pulse in Platform.Pulses])
        Comp = np.abs(np.expand_dims(FreqTrackers, axis=0)-np.expand_dims(FreqPulses, axis=1)) > self.FreqThreshold
        if np.sum(Comp) == 0:
            return False
        Id = np.argmax(Comp*np.expand_dims(LevelTrackers, axis=0), axis=0)
        return Id

    def RunPlatform(self, Pulse):
        Id = self.CheckCompatibility(Pulse)
        if Id is False:
            if len(self.Trackers) < self.NbTrackers:
                self.OpenTracker(Pulse)
            else:
                self.UpdateTracker(Id, Pulse)
        self.CheckAgeTrackers()