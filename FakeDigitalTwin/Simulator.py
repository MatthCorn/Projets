class DigitalTwin():
    def __init__(self, AntPulses):
        self.AntPulses = AntPulses
        self.CurrentPulses = []
        self.TimeId = -1
        self.LevelStartingTime = 0
        self.LevelEndingTime = 0

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
        self.LevelStartingTime = self.CurrentPulses[0]['TOA']

        # TOE : Time of ending
        TOEList = [el['TOA'] + el['LI'] for el in self.CurrentPulses]
        minTOE = min(TOEList)
        if minTOE < self.AntPulses[self.TimeId + 1]['TOA']:
            # un palier commence car une impulsion se termine
            self.LevelEndingTime = minTOE
            self.LevelProcessing()
            del(self.CurrentPulses[TOEList.index(minTOE)])
        else:
            # un palier commence car une impulsion arrive
            self.LevelEndingTime = self.AntPulses[self.TimeId + 1]['TOA']
            self.LevelProcessing()
            self.TimeId += 1
            self.CurrentPulses.append(self.AntPulses[self.TimeId])
        self.forward()

    def step(self):
        if self.CurrentPulses == []:
            self.LevelStartingTime = self.LevelEndingTime
            self.LevelEndingTime = self.AntPulses[self.TimeId + 1]['TOA']
            self.LevelProcessing()
            self.TimeId += 1
            self.CurrentPulses.append(self.AntPulses[self.TimeId])
        else:
            # TOE : Time of ending
            TOEList = [el['TOA'] + el['LI'] for el in self.CurrentPulses]
            minTOE = min(TOEList)
            if minTOE < self.AntPulses[self.TimeId + 1]['TOA']:
                # un palier commence car une impulsion se termine
                self.LevelEndingTime = minTOE
                self.LevelProcessing()
                del(self.CurrentPulses[TOEList.index(minTOE)])
            else:
                # un palier commence car une impulsion arrive
                self.LevelEndingTime = self.AntPulses[self.TimeId + 1]['TOA']
                self.LevelProcessing()
                self.TimeId += 1
                self.CurrentPulses.append(self.AntPulses[self.TimeId])
        self.forward()

    def termination(self):
        if self.CurrentPulses==[]:
            return None
        # TOE : Time of ending
        TOEList = [el['TOA'] + el['LI'] for el in self.CurrentPulses]
        minTOE = min(TOEList)

        # un palier commence car une impulsion se termine
        self.LevelEndingTime = minTOE
        self.LevelProcessing()
        del (self.CurrentPulses[TOEList.index(minTOE)])

        self.forward()

    def LevelProcessing(self):
        # ne pas utiliser self.TimeId sinon risque de bug
        print(self.CurrentPulses)
        None


if __name__=='__main__':
    AntP = [{'TOA': 5 * k, 'LI': k} for k in range(1, 10)]
    DT = DigitalTwin(AntP)
    DT.forward()