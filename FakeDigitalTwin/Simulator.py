from FakeDigitalTwin.Trackers import Tracker, Pulse
from FakeDigitalTwin.Platform import Platform, Processor
class DigitalTwin():
    def __init__(self, AntPulses, NbMaxTrackers=4):
        self.AntPulses = AntPulses
        self.Platform = Platform()
        self.LevelThreshold = 0
        self.TimeId = -1

        # Making trackers to follow the pulses through the platforms
        self.Trackers = []
        for i in range(NbMaxTrackers):
            self.Trackers.append(Tracker(Id=str(i), parent=self, MaxAge=5))

        self.Processor = Processor()
        self.PDWs = []

    def forward(self):
        if self.TimeId == -1:
            self.initialization()
        elif self.TimeId < len(self.AntPulses)-1:
            self.step()
        else:
            self.termination()

    def initialization(self):
        self.TimeId += 1
        self.Platform.AddPulse(self.AntPulses[self.TimeId])
        self.Platform.StartingTime = self.Platform.Pulses[0].TOA

        # TOE : Time of ending
        TOEList = [Pulse.TOA + Pulse.LI for Pulse in self.Platform.Pulses]
        minTOE = min(TOEList)
        if minTOE < self.AntPulses[self.TimeId + 1].TOA:
            # un palier commence car une impulsion se termine
            self.Platform.EndingTime = minTOE
            self.PlatformProcessing()
            self.Platform.DelPulse(TOEList.index(minTOE))
        else:
            # un palier commence car une impulsion arrive
            self.Platform.EndingTime = self.AntPulses[self.TimeId + 1].TOA
            self.PlatformProcessing()
            self.TimeId += 1
            self.Platform.AddPulse(self.AntPulses[self.TimeId])
        self.forward()

    def step(self):
        self.Platform.StartingTime = self.Platform.EndingTime
        if self.Platform.IsEmpty():
            self.Platform.EndingTime = self.AntPulses[self.TimeId + 1].TOA
            self.PlatformProcessing()
            self.TimeId += 1
            self.Platform.AddPulse(self.AntPulses[self.TimeId])
        else:
            # TOE : Time of ending
            TOEList = [Pulse.TOA + Pulse.LI for Pulse in self.Platform.Pulses]
            minTOE = min(TOEList)
            if minTOE < self.AntPulses[self.TimeId + 1].TOA:
                # un palier commence car une impulsion se termine
                self.Platform.EndingTime = minTOE
                self.PlatformProcessing()
                self.Platform.DelPulse(TOEList.index(minTOE))
            elif minTOE > self.AntPulses[self.TimeId + 1].TOA:
                # un palier commence car une impulsion arrive
                self.Platform.EndingTime = self.AntPulses[self.TimeId + 1].TOA
                self.PlatformProcessing()
                self.TimeId += 1
                self.Platform.AddPulse(self.AntPulses[self.TimeId])
            else :
                # un palier commence car une impulsion se termine ET une impulsion arrive EN MEME TEMPS
                self.Platform.EndingTime = minTOE
                self.PlatformProcessing()
                self.Platform.DelPulse(TOEList.index(minTOE))
                self.TimeId += 1
                self.Platform.AddPulse(self.AntPulses[self.TimeId])
        self.forward()

    def termination(self):
        if self.Platform.IsEmpty():
            self.PlatformProcessing()
            return None
        self.Platform.StartingTime = self.Platform.EndingTime
        # TOE : Time of ending
        TOEList = [Pulse.TOA + Pulse.LI for Pulse in self.Platform.Pulses]
        minTOE = min(TOEList)

        # un palier commence car une impulsion se termine
        self.Platform.EndingTime = minTOE
        self.PlatformProcessing()
        self.Platform.DelPulse(TOEList.index(minTOE))

        self.forward()

    def PlatformProcessing(self):
        # On laisse cette fonction le temps de tester
        self.Processor.RunPlatform(self.Platform, self.Trackers)

        # ne pas utiliser self.TimeId sinon risque de bug
        print('starting time :', self.Platform.StartingTime)
        print('curent pulses :', self.Platform.Pulses)
        print('visible pulses :', self.Platform.VisiblePulses)
        print('ending time :', self.Platform.EndingTime)
        print('trackers:', [el for el in self.Trackers if el.IsTaken])
        print('PDWs:', self.PDWs)
        print('\n')
        None


if __name__ == '__main__':
    import numpy as np
    import scipy as sp
    # AntP = [Pulse(TOA=1, LI=16, FreqStart=10, FreqEnd=12, Level=1), Pulse(TOA=7, LI=12, FreqStart=9, FreqEnd=6, Level=1)]
    # AntP = [Pulse(TOA=5*k, LI=k, FreqStart=np.random.randint(7, 13), FreqEnd=np.random.randint(7, 13), Level=5.5*np.random.random()) for k in range(4, 13)]

    # On se donne un scénario de 1000 unité de temps
    # Le temps de maintien max est de tmax unité de temps, on veut que 99% des impulsions soit moins longues
    tmax = sp.stats.gamma.ppf(0.99, a=2, scale=1)
    AntP = [Pulse(TOA=1000*np.random.random(), LI=np.random.gamma(shape=2, scale=1))]
    DT = DigitalTwin(AntP)
    DT.forward()
