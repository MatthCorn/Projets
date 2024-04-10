class Plateau():
    def __init__(self, StartingTime=0, EndingTime=0, Parent=None):
        self.Parent = Parent

        # List of the pulses on the plateau. They are sorted by time of departure
        self.Pulses = []

        # We built a second list where the pulses are sorted by level
        self.LevelPulses = []

        # List of the non-ambiguous detection after computation
        self.NAD = []

        self.StartingTime = StartingTime
        self.EndingTime = EndingTime

    # This function determines the time of the end of the current plateau
    def SetEndingTime(self):

        # Time of the next departure in the plateau
        try:
            NextDeparture = self.Pulses[0]
            ToD = NextDeparture.TOA + NextDeparture.LI
        except:
            ToD = float('inf')

        # Time of arrival of the upcoming pulse on the plateau
        try:
            NextArrival = self.Parent.AntPulses[0]
            ToA = NextArrival.TOA
        except:
            ToA = float('inf')

        # This date determinates the end of the plateau
        Tmin = min(ToD, ToA)

        # We use this date to end the computation on the plateau
        self.EndingTime = Tmin

    # This function aims to update the pulses to represent the next plateau
    def Next(self):

        # We update the plateau
        self.StartingTime = self.EndingTime

        # If StartingTime is inf, we are already on the last plateau
        if self.StartingTime == float('inf'):
            self.Parent.Running = False
        else:
            # We delete the pulses whose ToD are StartingTime (can be several)
            # This deletion is made on both self.Pulses and self.LevelPulses
            try:
                while self.Pulses[0].TOA + self.Pulses[0].LI == self.StartingTime:
                    Pulse = self.Pulses.pop(0)
                    # print('Deleting ', Pulse)
                    self.DelLevel(Pulse)
            except:
                pass

            # We add the pulses whose ToA are StartingTime (can be several)
            try:
                while self.Parent.AntPulses[0].TOA == self.StartingTime:
                    Pulse = self.Parent.AntPulses.pop(0)
                    # print('Adding ', Pulse)
                    self.AddPulse(Pulse)

            except:
                pass

        # We set the ending time of the plateau
        self.SetEndingTime()


    # This function adds the next pulse to the plateau
    def AddPulse(self, Pulse):
        # we are looking for the place of the new pulse in the list sorted by increasing ToD
        ToD = Pulse.TOA + Pulse.LI
        Id = -1


        for i in reversed(range(len(self.Pulses))):
            if self.Pulses[i].TOA + self.Pulses[i].LI < ToD:
                Id = i
                break

        self.Pulses.insert(Id + 1, Pulse)

        Id = -1
        # we are looking for the place of the new pulse in the list sorted by increasing level
        Level = Pulse.Level
        for i in reversed(range(len(self.LevelPulses))):
            if self.LevelPulses[i].Level < Level:
                Id = i
                break

        self.LevelPulses.insert(Id + 1, Pulse)

    # This function deletes a pulse from self.LevelPulses, using dichotomy to locate it
    def DelLevel(self, Pulse):
        a = 0
        b = len(self.LevelPulses)
        while a != b:
            i = int((a+b)/2)
            if self.LevelPulses[i].Level < Pulse.Level:
                a = i+1
            else:
                b = i

        if Pulse is not self.LevelPulses[a]:
            # print('There are several pulses with the same level, dichotomy impossible')
            self.LevelPulses.remove(Pulse)

        else:
            del(self.LevelPulses[a])

    # This function tells if the plateau is empty
    def IsEmpty(self):
        return self.Pulses == []
