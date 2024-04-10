class Meter():
    def __init__(self, Id, Parent=None):
        self.Id = Id
        self.Parent = Parent
        self.IsOpen = False
        self.InMaintenance = False

    def Link(self, Pulse, Ts, Te):
        self.LastPulse = Pulse

        # if the meter is free, we open it
        if not self.IsOpen:
            self.Open(Ts, Te)

        # else the meter and the pulse correlated, we update the meter
        else:
            self.Update(Te)


    def Open(self, Ts, Te):
        self.Ts = Ts
        self.Tc = Te

        # if the meter opens before the last pulse (possible in CW closing)
        if Ts < self.LastPulse.TOA:
            self.FreqStart = self.FreqCur
        else:
            self.FreqStart = self.LastPulse.GetFreq(Ts)

        self.FreqCur = self.LastPulse.GetFreq(Te)

        self.FreqMin = self.FreqStart
        self.FreqMax = self.FreqStart

        self.Level = self.LastPulse.Level

        self.InMaintenance = True
        self.IsOpen = True

    def Update(self, Te):
        self.Tc = Te

        self.FreqCur = self.LastPulse.GetFreq(Te)

        self.InMaintenance = True

    def Maintenance(self):
        Param = self.Parent.Param

        Tcw = self.Ts + Param['Durée max impulsion']
        # If the meter is opened for too long, we update and release it

        if Tcw < self.Tc:
            Te, self.Tc = self.Tc, Tcw

            # If the rupture occurs between two pulses, we keep the current frequency
            if Tcw > self.LastPulse.TOA:
                self.FreqCur = self.LastPulse.GetFreq(self.Tc)

            self.FreqMin = min(self.FreqMin, self.FreqCur)
            self.FreqMax = max(self.FreqCur, self.FreqCur)

            self.Release()

            # The meter reopens with the same pulse and goes back to maintenance
            self.Open(Tcw, Te)

        # If not, we update it and leave maintenance
        else:
            self.FreqCur = self.LastPulse.GetFreq(self.Tc)
            self.FreqMin = min(self.FreqMin, self.FreqCur)
            self.FreqMax = max(self.FreqCur, self.FreqCur)
            self.InMaintenance = False

    # This function releases a PDW and store it, the meter is then free to use
    def Release(self, Tp=0):
        self.Parent.AddPDW(self.Emission(), Tp)

        self.InMaintenance = False
        self.IsOpen = False

    def Emission(self):
        PDW = {'TOA': self.Ts, 'LI': self.Tc - self.Ts, 'Level': self.Level, 'FreqMin': self.FreqMin, 'FreqMax': self.FreqMax}
        return PDW

    def __repr__(self):
        return '(Meter : Taken={} ; TOA={} ; TOD={} ; Freq={}; Meter Id={})'.format(self.IsOpen, round(self.Ts, 3), round(self.Tc, 3),
                                                                                        round(self.FreqCur, 3), self.Id)
