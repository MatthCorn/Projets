class Tracker():
    def __init__(self, Id, MaxAge, HoldingTime=2, parent=None):
        self.id = Id
        self.MaxAge = MaxAge
        self.HoldingTime = HoldingTime
        self.parent = parent
        self.IsTaken = False
        self.FreqCur = -float('inf')
        self.Level = 0
        self.TOA = 0
        self.TOE = -1
        self.flags = []
        self.Histogram = []

    def Open(self, Id, TOA=None, flag=[]):
        if self.IsTaken:
            return False
        # print('Tracker no. {} opened'.format(self.id))
        # LastPulseId permet de réouvrir correctement un mesureur en cas de CW
        self.LastPulseId = Id
        self.Histogram = [self.parent.Platform.Pulses[Id].Id]
        # Id est l'indice de l'impulsion qui nous intéresse dans la liste des impulsions du palier

        self.IsTaken = True
        self.flags = set(flag)
        self.TOA = self.parent.Platform.StartingTime
        if TOA is not None:
            self.TOA = TOA
        if self.TOE == self.TOA:
            # Si le mesureur vient d'être libéré (donc il y avait saturation des mesureurs)
            # On ajoute le flag "troncature avant"
            self.flags.add('TroncAv')
        self.TOE = self.parent.Platform.EndingTime
        self.FreqMin = self.parent.Platform.Pulses[Id].GetFreq(self.TOA)
        self.FreqCur = self.parent.Platform.Pulses[Id].GetFreq(self.TOE)
        self.FreqMax = self.parent.Platform.Pulses[Id].GetFreq(self.TOA)
        self.Level = self.parent.Platform.Pulses[Id].Level
        return True

    def Release(self, flag=None):
        if flag is not None:
            self.flags.add(flag)
        self.parent.PDWs.append(self.Emission())
        self.IsTaken = False

    def Update(self, Id):
        # print('update tracker no.{} with Id{}'.format(self.id, Id))
        self.LastPulseId = Id
        platform = self.parent.Platform
        self.Histogram.append(platform.Pulses[Id].Id)
        self.TOE = platform.EndingTime
        CurrentFreq = platform.Pulses[Id].GetFreq(self.TOE)
        self.FreqCur = CurrentFreq

    def Check(self, platform):
        # Age du mesureur
        Age = self.TOE - self.TOA

        if platform.IsEmpty():
            # Si la platform est vide, tous les mesureurs sont fermés
            self.Release()

        elif Age > self.MaxAge:
            # Si le mesureur est ouvert depuis trop longtemps
            self.TOE = self.TOA + self.MaxAge
            TOA_CW = self.TOE

            # On regarde si la dernière impulsion arrive sur la rupture de suivi du mesureur
            # Il se peut que non si la trace de l'impulsion était perdue sur le parlier de la date d'arrêt de suivi du mesureur
            Id = self.LastPulseId
            LastPulse = platform.Pulses[Id]
            if self.TOE > LastPulse.TOA:
                # Si la dernière impulsion arrive sur la rupture de suivi du mesureur
                # On met à jour les fréquences min et max
                CurrentFreq = LastPulse.GetFreq(self.TOE)
                self.FreqMin = min(self.FreqMin, CurrentFreq)
                self.FreqMax = max(self.FreqMax, CurrentFreq)

                # On émet un PDW avec un flag "CW"
                self.Release(flag='CW')
                # Puis on réouvre le mesureur avec la même impulsion
                self.Open(Id=Id, TOA=TOA_CW, flag=['CW'])
                self.Check(platform)
            else:
                # Sinon, le mesureur est en fait coupé avant l'arrivé de la dernière impulsion
                # D'abord on émet un PDW puis on s'occupe de la nouvelle impulsion
                Id = self.LastPulseId

                # On émet un PDW avec un flag 'CW'
                self.Release(flag='CW')
                # On réouvre le mesureur et on le laisse prendre les infos de la nouvelle impulsion
                self.Open(Id=Id, flag=['CW'])
                # On change la TOA du mesureur (une fois que les fréquences sont initialisées par Open())
                self.TOA = TOA_CW
                self.Check(platform)




        elif platform.EndingTime - self.TOE > self.HoldingTime:
            # Si le mesureur a perdu la trace de son impulsion
            self.Release()
            self.TOE = platform.EndingTime

        elif platform.EndingTime == self.TOE:
            # Sinon, si le mesureur vient effectivement de suivre une impulsion sur le palier, on met à jour les fréquences min et max
            Id = self.LastPulseId
            CurrentFreq = platform.Pulses[Id].GetFreq(self.TOE)
            self.FreqMin = min(self.FreqMin, CurrentFreq)
            self.FreqMax = max(self.FreqMax, CurrentFreq)

    def Emission(self):
        PDW = {'TOA': self.TOA, 'LI': self.TOE - self.TOA, 'Level': self.Level, 'FreqMin': self.FreqMin, 'FreqMax': self.FreqMax, 'flags': self.flags}
        return PDW

    def __repr__(self):
        return '(Tracker : Taken={} ; TOA={} ; TOE={} ; Histogram={} ; Freq={} ; Tracker Id={})'.format(self.IsTaken, round(self.TOA, 3), round(self.TOE, 3),
                                                                                                        self.Histogram, round(self.FreqCur, 3), self.id)
class Pulse():
    def __init__(self, FreqStart=0, FreqEnd=0, TOA=0, LI=0, Level=0, Id=0):
        self.FreqStart = FreqStart
        self.FreqEnd = FreqEnd
        self.TOA = TOA
        self.LI = LI
        self.Level = Level
        self.Id = Id

    def GetFreq(self, t):
        TOA, TOE = self.TOA, self.TOA+self.LI
        if (t < TOA) or (t > TOE):
            raise ValueError('"t" is not in the pulse emission time interval')
        return (self.FreqStart*(TOE-t) + self.FreqEnd*(t-TOA))/self.LI

    def __str__(self):
        return 'Pulse(TOA={}, LI={}, Level={}, FreqStart={}, FreqEnd={}, Id={})'.format(self.TOA, self.LI, self.Level, self.FreqStart, self.FreqEnd, self.Id)

    def __repr__(self):
        return 'Pulse(TOA={}, LI={}, Level={}, FreqStart={}, FreqEnd={}, Id={})'.format(self.TOA, self.LI, self.Level, self.FreqStart, self.FreqEnd, self.Id)
