import numpy as np

class Processor():
    def __init__(self, Parent=None):
        self.Param = Parent.Param
        self.Mono, self.HarmoTreat, self.IMTreat = False, False, False
        self.RejectHThreshold, self.RejectIMThreshold = 0, 0

    # This function determines which pulse is visible or annoying based on its level
    # It also determines if mono-signal, harmonic or inter-modulation treatment will be needed
    # It returns those information in addition of thresholds for the treatment
    def TrackTreatment(self, Plateau):
        self.Mono, self.HarmoTreat, self.IMTreat = False, False, False
        self.RejectHThreshold, self.RejectIMThreshold = 0, 0

        # We determine whether there will be a mono-signal, harmonic or inter-modulation treatment

        try:
            SecondPulse, FirstPulse = Plateau.LevelPulses[-2:]

            # We determine if the plateau is mono-signal
            if FirstPulse.Level > self.Param['Seuil_mono']:
                self.Mono = True
                self.VisibleId = len(Plateau.Pulses)-1
                self.InteractingId = len(Plateau.Pulses)-1
                return

            # We determine if there is a need of a harmonic or an inter-modulation processing
            self.HarmoTreat = FirstPulse.Level > self.Param['Seuil_harmo']
            if self.HarmoTreat:
                self.RejectHThreshold = self.Param['Seuil_sensi_traitement'] + 2 * (FirstPulse.Level - self.Param['Seuil_harmo'])

            self.IMTreat = SecondPulse.Level > self.Param['Seuil_IM'] - (FirstPulse.Level - self.Param['Seuil_IM'])
            if self.IMTreat:
                self.RejectIMThreshold = self.Param['Seuil_sensi_traitement'] + 2 * (FirstPulse.Level - self.Param['Seuil_IM'])
        except:
            pass

        # This first index informs on the first pulse that can interact with other (annoying or visible pulses)
        self.InteractingId = len(Plateau.Pulses)

        # This second index informs on the first pulse that can be seen (only visible pulses)
        self.VisibleId = len(Plateau.Pulses)

        for pulse in list(reversed(Plateau.LevelPulses)):
            ThresholdVisibility = self.Param['Seuil_sensi']
            ThresholdInteracting = self.Param['Seuil_sensi'] - self.Param['Contraste_geneur']
            if self.IMTreat:
                ThresholdVisibility = self.RejectIMThreshold

            if pulse.Level > ThresholdInteracting:
                self.InteractingId -= 1
            else:
                break

            if pulse.Level > ThresholdVisibility:
                self.VisibleId -= 1


    # This function computes the interaction between pulses on a track with frequency Fe
    # Its outputs inform on the elementary detections on this track
    def TrackDetection(self, Plateau, Fe):
        if Plateau.IsEmpty():
            return []

        FoldFun = lambda Freq: min(Freq - (Freq // Fe) * Fe, -Freq - (-Freq // Fe) * Fe)
        CanalFun = lambda Freq: round(self.Param['Nint'] * Freq / Fe)

        Ts = Plateau.StartingTime

        # We compute the frequency of the folded spectrum between 0 et Fe/2 for each interacting pulse and store the corresponding canal
        CanalList = []

        for Pulse in Plateau.LevelPulses[self.InteractingId:]:
            # We compute the frequency of the folded spectrum between 0 et Fe/2
            PulseFreq = Pulse.GetFreq(Ts)
            FoldFreq = FoldFun(PulseFreq)

            # We compute the canal corresponding
            Canal = CanalFun(FoldFreq)
            CanalList.append(Canal)

        # In case of harmonic processing
        if self.HarmoTreat:
            # We compute the folded frequency of the 2nd and 3rd harmonic of the brightest pulse
            # Those harmonics can mask other pulses
            PulseFreq = Plateau.LevelPulses[-1].GetFreq(Ts)
            FoldH2 = FoldFun(2*PulseFreq)
            FoldH3 = FoldFun(3*PulseFreq)

        # We built a list informing of which visible pulse is in fact masked or polluted
        MaskedList = []
        PollutedList = []

        for i in range(self.VisibleId, len(Plateau.Pulses)):
            VisiblePulse = Plateau.LevelPulses[i]
            Polluted = False
            Masked = False

            VisibleCanal = CanalList[i - self.InteractingId]

            # First and last canals are masked
            if VisibleCanal < self.Param['M1_aveugle'] or VisibleCanal > self.Param['Nint']/2 - self.Param['M2_aveugle']:
                PollutedList.append(True)
                MaskedList.append(True)
                continue

            # We check for the harmonic interaction
            if self.HarmoTreat:
                if abs(VisibleCanal - CanalFun(FoldH2)) < 2 or abs(VisibleCanal - CanalFun(FoldH3)) < 2:
                    if VisiblePulse.Level < self.RejectHThreshold:
                        PollutedList.append(True)
                        MaskedList.append(True)
                        continue

            # We check for interaction with other pulses by decreasing level
            for j in reversed(range(self.InteractingId, len(Plateau.Pulses))):
                InteractingPulse = Plateau.LevelPulses[j]
                InteractingCanal = CanalList[j - self.InteractingId]

                if InteractingPulse is VisiblePulse:
                    continue

                # if a pulse level is too low, the following pulse levels in the loop will be too low
                if VisiblePulse.Level - InteractingPulse.Level > self.Param['Contraste_geneur_2']:
                    Masked = False
                    break

                else:
                    # if pulse canals are close, they will interact
                    if abs(VisibleCanal - InteractingCanal) < self.Param['M_local']:
                        if VisiblePulse.Level < InteractingPulse.Level:
                            Polluted = True
                            Masked = True
                            break
                        else:
                            Polluted = True

            PollutedList.append(Polluted)
            MaskedList.append(Masked)

        # We make sure there is not more than "N_DetEl" pulses transmitted
        # It has to be the strongest pulses in terms of level
        NDetEl = 0
        for i in reversed(range(len(MaskedList))):
            if not MaskedList[i]:
                if NDetEl == self.Param['N_DetEl']:
                    MaskedList[i] = True
                    PollutedList[i] = True
                else:
                    NDetEl += 1

        return MaskedList, PollutedList

    # This function takes the elementary detections of each canal and fuse them into
    # NAD, simulating the behavior of the frequency ambiguity removal module
    def FreqAmbRemoval(self, Plateau):
        FeList = self.Param['Fe_List']
        # We create 2 scores which will determine whether a pulse is a NAD or not
        # Both of them must stay above 0 for a pulse to be a NAD
        ScoreMask = 1
        ScorePollution = 1

        self.TrackTreatment(Plateau)

        if self.Mono:
            return Plateau.LevelPulses[-1:]

        for Fe in FeList:
            MaskedList, PollutedList = self.TrackDetection(Plateau, Fe)
            ScoreMask -= np.array(MaskedList) / 2
            ScorePollution -= np.array(PollutedList) / 2
        # IsNAD = np.multiply(ScorePollution > 0, ScoreMask > 0)
        IsNAD = ScoreMask > 0
        # We build the list of NAD and return it
        NVisible = len(IsNAD)
        VisiblePulses = np.array(Plateau.LevelPulses[-NVisible:])
        NAD = VisiblePulses[IsNAD]
        return NAD

    # This function determines which NAD goes with which meter
    def LinkMeterNAD(self, NAD, Meters, Plateau):

        Ts = Plateau.StartingTime
        Te = Plateau.EndingTime

        OpenedMeters = [meter for meter in Meters if meter.IsOpen]
        FreeMeters = [meter for meter in Meters if not meter.IsOpen]

        # The list LostNAD contains the NAD that have not been linked to a meter
        # because there is no free meter left
        LostNAD = []

        # Each NAD can be linked to an opened meter, prioritising high level pulse
        for Pulse in reversed(NAD):

            # We determine which meter match the most with the NAD, prioritising high level meter
            BestMatchingId = None
            for i in range(len(OpenedMeters)):
                meter = OpenedMeters[i]
                if abs(Pulse.GetFreq(Ts) - meter.FreqCur) < self.Param['Seuil_ecart_freq']:
                    try:
                        if meter.Level > OpenedMeters[BestMatchingId].Level:
                            BestMatchingId = i
                    except:
                        BestMatchingId = i

            # We link the NAD to the meter it opens or updates
            if BestMatchingId is None:
                try:
                    meter = FreeMeters.pop()
                    meter.Link(Pulse, Ts, Te)
                except:
                    LostNAD.append(Pulse)

            else:
                meter = OpenedMeters.pop(BestMatchingId)
                meter.Link(Pulse, Ts, Te)

        # We try to link the lost NAD to meters that will be freed during the plateau
        OpenedMeters.sort(key=lambda meter: meter.Tc)
        for meter in OpenedMeters:
            Tp = meter.Tc + self.Param['Duree_maintien_max']
            # If the meter did lose its pulse
            if Tp < Te:
                # We release it
                meter.Release(Tp)

                # We link the brightest NAD to the meter
                if len(LostNAD) > 0:
                    meter.Link(LostNAD.pop(0), Tp, Te)

            else:
                break
