from FakeDigitalTwin.Trackers import Tracker
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
            self.Trackers.append(Tracker(Id=str(i), parent=self))

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
        self.Platform.StartingTime = self.Platform.Pulses[0]['TOA']

        # TOE : Time of ending
        TOEList = [el['TOA'] + el['LI'] for el in self.Platform.Pulses]
        minTOE = min(TOEList)
        if minTOE < self.AntPulses[self.TimeId + 1]['TOA']:
            # un palier commence car une impulsion se termine
            self.Platform.EndingTime = minTOE
            self.PlatformProcessing()
            self.Platform.DelPulse(TOEList.index(minTOE))
        else:
            # un palier commence car une impulsion arrive
            self.Platform.EndingTime = self.AntPulses[self.TimeId + 1]['TOA']
            self.PlatformProcessing()
            self.TimeId += 1
            self.Platform.AddPulse(self.AntPulses[self.TimeId])
        self.forward()

    def step(self):
        self.Platform.StartingTime = self.Platform.EndingTime
        if self.Platform.IsEmpty():
            self.Platform.EndingTime = self.AntPulses[self.TimeId + 1]['TOA']
            self.PlatformProcessing()
            self.TimeId += 1
            self.Platform.AddPulse(self.AntPulses[self.TimeId])
        else:
            # TOE : Time of ending
            TOEList = [el['TOA'] + el['LI'] for el in self.Platform.Pulses]
            minTOE = min(TOEList)
            if minTOE < self.AntPulses[self.TimeId + 1]['TOA']:
                # un palier commence car une impulsion se termine
                self.Platform.EndingTime = minTOE
                self.PlatformProcessing()
                self.Platform.DelPulse(TOEList.index(minTOE))
            elif minTOE > self.AntPulses[self.TimeId + 1]['TOA']:
                # un palier commence car une impulsion arrive
                self.Platform.EndingTime = self.AntPulses[self.TimeId + 1]['TOA']
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
            return None
        self.Platform.StartingTime = self.Platform.EndingTime
        # TOE : Time of ending
        TOEList = [el['TOA'] + el['LI'] for el in self.Platform.Pulses]
        minTOE = min(TOEList)

        # un palier commence car une impulsion se termine
        self.Platform.EndingTime = minTOE
        self.PlatformProcessing()
        self.Platform.DelPulse(TOEList.index(minTOE))

        self.forward()

    def PlatformProcessing(self):
        # ne pas utiliser self.TimeId sinon risque de bug
        print('starting time :', self.Platform.StartingTime)
        print('curent pulses :', self.Platform.Pulses)
        print('ending time :', self.Platform.EndingTime)
        print('\n')
        None


if __name__=='__main__':
    AntP = [{'TOA': 5 * k, 'LI': k} for k in range(1, 10)]
    DT = DigitalTwin(AntP)
    DT.forward()