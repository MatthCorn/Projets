from FakeDigitalTwin.Trackers import Tracker
from FakeDigitalTwin.Platform import Platform, Processor
class DigitalTwin():
    def __init__(self, AntPulses, NbMaxTrackers=4):
        self.AntPulses = AntPulses
        self.CurrentPulses = []
        self.LevelThreshold = 0
        self.TimeId = -1
        self.PlatformStartingTime = 0
        self.PlatformEndingTime = 0

        # Making trackers to follow the pulses through the platforms
        self.FreeTrackers = []
        for i in range(NbMaxTrackers):
            self.FreeTrackers.append(Tracker(id=str(i), parent=self))
        self.TakenTrackers = []

    def forward(self):
        if self.TimeId == -1:
            self.initialization()
        elif self.TimeId < len(self.AntPulses)-1:
            self.step()
        else:
            self.termination()

    def initialization(self):
        self.TimeId += 1
        self.CurrentPulses.append(self.AntPulses[self.TimeId])
        self.PlatformStartingTime = self.CurrentPulses[0]['TOA']

        # TOE : Time of ending
        TOEList = [el['TOA'] + el['LI'] for el in self.CurrentPulses]
        minTOE = min(TOEList)
        if minTOE < self.AntPulses[self.TimeId + 1]['TOA']:
            # un palier commence car une impulsion se termine
            self.PlatformEndingTime = minTOE
            self.PlatformProcessing()
            del(self.CurrentPulses[TOEList.index(minTOE)])
        else:
            # un palier commence car une impulsion arrive
            self.PlatformEndingTime = self.AntPulses[self.TimeId + 1]['TOA']
            self.PlatformProcessing()
            self.TimeId += 1
            self.CurrentPulses.append(self.AntPulses[self.TimeId])
        self.forward()

    def step(self):
        self.PlatformStartingTime = self.PlatformEndingTime
        if self.CurrentPulses == []:
            self.PlatformEndingTime = self.AntPulses[self.TimeId + 1]['TOA']
            self.PlatformProcessing()
            self.TimeId += 1
            self.CurrentPulses.append(self.AntPulses[self.TimeId])
        else:
            # TOE : Time of ending
            TOEList = [el['TOA'] + el['LI'] for el in self.CurrentPulses]
            minTOE = min(TOEList)
            if minTOE < self.AntPulses[self.TimeId + 1]['TOA']:
                # un palier commence car une impulsion se termine
                self.PlatformEndingTime = minTOE
                self.PlatformProcessing()
                del(self.CurrentPulses[TOEList.index(minTOE)])
            elif minTOE > self.AntPulses[self.TimeId + 1]['TOA']:
                # un palier commence car une impulsion arrive
                self.PlatformEndingTime = self.AntPulses[self.TimeId + 1]['TOA']
                self.PlatformProcessing()
                self.TimeId += 1
                self.CurrentPulses.append(self.AntPulses[self.TimeId])
            else :
                # un palier commence car une impulsion se termine ET une impulsion arrive EN MEME TEMPS
                self.PlatformEndingTime = minTOE
                self.PlatformProcessing()
                del (self.CurrentPulses[TOEList.index(minTOE)])
                self.TimeId += 1
                self.CurrentPulses.append(self.AntPulses[self.TimeId])
        self.forward()

    def termination(self):
        if self.CurrentPulses==[]:
            return None
        self.PlatformStartingTime = self.PlatformEndingTime
        # TOE : Time of ending
        TOEList = [el['TOA'] + el['LI'] for el in self.CurrentPulses]
        minTOE = min(TOEList)

        # un palier commence car une impulsion se termine
        self.PlatformEndingTime = minTOE
        self.PlatformProcessing()
        del (self.CurrentPulses[TOEList.index(minTOE)])

        self.forward()

    def PlatformProcessing(self):
        # ne pas utiliser self.TimeId sinon risque de bug
        print('starting time :', self.PlatformStartingTime)
        print('curent pulses :', self.CurrentPulses)
        print('ending time :', self.PlatformEndingTime)
        print('\n')
        None


if __name__=='__main__':
    AntP = [{'TOA': 5 * k, 'LI': k} for k in range(1, 10)]
    DT = DigitalTwin(AntP)
    DT.forward()