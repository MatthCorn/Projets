from FakeDigitalTwin.Trackers import Tracker, Pulse
from FakeDigitalTwin.Platform import Platform, Processor
class DigitalTwin():
    def __init__(self, NbMaxTrackers=4, FreqThreshold=1.5, Fe=500, MaxAgeTracker=5, FreqSensibility=1, SaturationThreshold=5, HoldingTime=2):
        self.Platform = Platform()
        self.TimeId = -1

        # Making trackers to follow the pulses through the platforms
        self.Trackers = []
        for i in range(NbMaxTrackers):
            self.Trackers.append(Tracker(Id=str(i), parent=self, MaxAge=MaxAgeTracker, HoldingTime=HoldingTime))

        self.Processor = Processor(FreqThreshold=FreqThreshold, Fe=Fe, FreqSensibility=FreqSensibility, SaturationThreshold=SaturationThreshold)
        self.PDWs = []

    def forward(self, AntPulses=None):
        if AntPulses is not None:
            self.AntPulses = AntPulses
        while self.TimeId < len(self.AntPulses):
            if self.TimeId == -1:
                self.initialization()
            elif self.TimeId < len(self.AntPulses) - 1:
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
            # un palier commence car une impulsion se termine (il ne peut pas y en avoir plusieurs)
            self.Platform.EndingTime = minTOE
            self.PlatformProcessing()
            self.Platform.DelPulse(TOEList.index(minTOE))
        else:
            # un palier commence car une ou plusieurs impulsions arrivent
            self.Platform.EndingTime = self.AntPulses[self.TimeId + 1].TOA
            self.PlatformProcessing()
            UpcommingTOA = self.AntPulses[self.TimeId + 1].TOA
            while self.AntPulses[self.TimeId + 1].TOA == UpcommingTOA:
                self.TimeId += 1
                self.Platform.AddPulse(self.AntPulses[self.TimeId])
                if self.TimeId == len(self.AntPulses) - 1:
                    # on s'arrête si on a ajouté toutes les impulsions
                    break
        return None

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
                # un palier commence car une ou plusieurs impulsions se terminent
                self.Platform.EndingTime = minTOE
                self.PlatformProcessing()
                Index = [i for i, TOE in enumerate(TOEList) if TOE == minTOE]
                self.Platform.DelPulse(Index)
            elif minTOE > self.AntPulses[self.TimeId + 1].TOA:
                # un palier commence car une ou plusieurs impulsions arrivent
                self.Platform.EndingTime = self.AntPulses[self.TimeId + 1].TOA
                self.PlatformProcessing()
                UpcommingTOA = self.AntPulses[self.TimeId + 1].TOA
                while self.AntPulses[self.TimeId + 1].TOA == UpcommingTOA:
                    self.TimeId += 1
                    self.Platform.AddPulse(self.AntPulses[self.TimeId])
                    if self.TimeId == len(self.AntPulses) - 1:
                        # on s'arrête si on a ajouté toutes les impulsions
                        break
            else :
                # un palier commence car une ou plusieurs impulsion se terminent ET une impulsion ou plusieurs arrivent EN MEME TEMPS
                self.Platform.EndingTime = minTOE
                self.PlatformProcessing()
                # On supprime les impulsions qui se terminent
                Index = [i for i, TOE in enumerate(TOEList) if TOE == minTOE]
                self.Platform.DelPulse(Index)

                # On ajoute les impulsions qui arrivent
                while self.AntPulses[self.TimeId + 1].TOA == minTOE:
                    self.TimeId += 1
                    self.Platform.AddPulse(self.AntPulses[self.TimeId])
                    if self.TimeId == len(self.AntPulses) - 1:
                        # on s'arrête si on a ajouté toutes les impulsions
                        break
        return None
    def termination(self):
        if self.Platform.IsEmpty():
            self.TimeId += 1
            self.PlatformProcessing()
            return None
        self.Platform.StartingTime = self.Platform.EndingTime
        # TOE : Time of ending
        TOEList = [Pulse.TOA + Pulse.LI for Pulse in self.Platform.Pulses]
        minTOE = min(TOEList)

        # un palier commence car une ou plusieurs impulsions se terminent
        self.Platform.EndingTime = minTOE
        self.PlatformProcessing()
        Index = [i for i, TOE in enumerate(TOEList) if TOE == minTOE]
        self.Platform.DelPulse(Index)
        return None
    def PlatformProcessing(self):
        self.Processor.RunPlatform(self.Platform, self.Trackers)

        # print('\n')
        # print('starting time :', self.Platform.StartingTime)
        # print('\n')
        # print('curent pulses :', self.Platform.Pulses)
        # print('\n')
        # print('visible pulses :', self.Platform.VisiblePulses)
        # print('\n')
        # print('ending time :', self.Platform.EndingTime)
        # print('\n')
        # print('trackers:', [el for el in self.Trackers if el.IsTaken])
        # print('\n')
        # print('PDWs:', self.PDWs)
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        # print('\n')
        # print('\n')
        # print('\n')


if __name__ == '__main__':
    import numpy as np
    # AntP = [Pulse(TOA=1, LI=16, FreqStart=10, FreqEnd=12, Level=1), Pulse(TOA=7, LI=12, FreqStart=9, FreqEnd=6, Level=1)]
    AntP = [Pulse(TOA=5*k, LI=k, FreqStart=np.random.randint(7, 13), FreqEnd=np.random.randint(7, 13), Level=5.5*np.random.random()) for k in range(4, 13)]

    DT = DigitalTwin()
    DT.forward(AntPulses=AntP)

