from FakeDigitalTwin.Meter import Meter
from FakeDigitalTwin.Pulse import Pulse
from FakeDigitalTwin.Plateau import Plateau
from FakeDigitalTwin.Processor import Processor
from time import time

class DigitalTwin():
    def __init__(self, Param):
        self.Param = Param
        # Making meters to follow the pulses through the plateaus, all initially free
        self.Meters = []
        for i in range(Param['N mesureurs max']):
            self.Meters.append(Meter(Id=str(i), Parent=self))

        # We use only one object to represente the plateaus, this object is updated to represent a change of plateau
        self.Plateau = Plateau(Parent=self)

        self.Processor = Processor(Parent=self)
        self.PDWs = []

        self.TrackingCounter = 0
        self.PlateauCounter = 0
        self.PlateauUpdateCounter = 0
        self.NADCounter = 0
        self.NbPulseInPlateau = 0

    def forward(self, AntPulses=None):
        if AntPulses is not None:
            self.AntPulses = AntPulses.copy()

        self.Running = True

        # This loop is in charge of making the operation on each Plateau
        while self.Running:
            self.PlateauCounter += 1
            self.PlateauProcessing()

        self.NbPulseInPlateau = self.NbPulseInPlateau/self.PlateauCounter
        self.PlateauUpdateCounter = self.PlateauUpdateCounter/self.PlateauCounter
        self.TrackingCounter = self.TrackingCounter/self.PlateauCounter
        self.NADCounter = self.NADCounter/self.PlateauCounter

        return self.PDWs

    def PlateauProcessing(self):

        # We load the pulses of the upcoming plateau
        t = time()
        self.Plateau.Next()
        self.PlateauUpdateCounter += time() - t

        self.NbPulseInPlateau += len(self.Plateau.Pulses)

        # print('\n')
        # print('starting time :', self.Plateau.StartingTime)
        # print('\n')
        # print('current pulses :', self.Plateau.Pulses)
        # print('\n')
        # print('ending time :', self.Plateau.EndingTime)
        # print('\n')

        # If the plateau is empty
        if self.Plateau.IsEmpty():
            # we release all opened meters
            t = time()
            for meter in self.Meters:
                if meter.IsOpen:
                    meter.Release(self.Plateau.StartingTime)
            self.TrackingCounter += time() - t

        else:
            # We compute the NAD on the current plateau
            t = time()
            NAD = self.Processor.FreqAmbRemoval(self.Plateau, FeList=self.Param['Fe list'])
            self.NADCounter += time() - t

            # We link the NAD to the meters and start updating them
            t = time()
            self.Processor.LinkMeterNAD(NAD, self.Meters, self.Plateau)

            # We end updating the meters by making the maintenance
            for meter in self.Meters:
                if meter.InMaintenance:
                    meter.Maintenance()
            self.TrackingCounter += time() - t

        # print('meters:', [el for el in self.Meters if el.IsOpen])
        # print('\n')
        # print('meters Ids:', [el.Id for el in self.Meters if el.IsOpen])
        # print('\n')
        # print('PDWs:', self.PDWs)
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        # print('\n')
        # print('\n')
        # print('\n')

    def AddPDW(self, PDW, Tp=0):
        if self.Param['PDW triés']:
            TpList = [x[1] for x in self.PDWs]

            # We're looking for the place of the new pulse in self.PDWs emitted at the time Tp
            # It is more likely to be added at the end of self.PDWs
            for i in reversed(range(len(self.PDWs))):
                if TpList[i] < Tp:
                    self.PDWs.append([PDW, Tp])
                    break

        else:
            self.PDWs.append(PDW)


if __name__ == '__main__':
    import numpy as np
    # AntP = [Pulse(TOA=1, LI=16, FreqStart=10, FreqEnd=12, Level=1), Pulse(TOA=7, LI=12, FreqStart=9, FreqEnd=6, Level=1)]
    AntP = [Pulse(TOA=5*k, LI=k, FreqStart=np.random.randint(7, 13), FreqEnd=np.random.randint(7, 13), Level=5.5*np.random.random()) for k in range(4, 13)]

    DT = DigitalTwin()
    DT.forward(AntPulses=AntP)

