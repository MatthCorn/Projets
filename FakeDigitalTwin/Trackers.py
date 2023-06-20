import numpy as np

class Trackers():
    def __init__(self, NbTrackers):
        self.NbTrackers = NbTrackers
        self.Trackers = []

    def SetParam(self, Fe = 5000):
        self.Fe = Fe

    def Compatibility(self, Pulse):
        FreqTrackers = np.array([tracker['FREQ'] for tracker in self.Trackers])
        FreqPulse = Pulse['FREQ']
        Comp = np.abs(FreqTrackers-FreqPulse
