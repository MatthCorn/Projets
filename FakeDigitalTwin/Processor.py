import numpy as np

class Processor():
    def __init__(self, Parent=None):
        self.Param = Parent.Param

    # This function determines which pulse is visible or annoying based on its level
    # It also determines if mono-signal, harmonic or inter-modulation treatment will be needed
    # It returns those information in addition of thresholds for the treatment
    def TrackTreatment(self, Plateau):

        # We determine whether there will be a mono-signal, harmonic or inter-modulation treatment
        Mono, HarmoTreat, IMTreat = False, False, False
        RejectHThreshold, RejectIMThreshold = 0, 0
        try:
            FirstPulse, SecondPulse = Plateau.LevelPulses[-2:]

            # We determine if the plateau is mono-signal
            if FirstPulse.Level > self.Param['Seuil mono']:
                Mono = True

            # We determine if there is a need of a harmonic or an inter-modulation processing
            HarmoTreat = FirstPulse.Level > self.Param['Seuil harmo']
            if HarmoTreat:
                RejectHThreshold = self.Param['Seuil sensi traitement'] + 2 * (FirstPulse.Level - self.Param['Seuil harmo'])

            IMTreat = SecondPulse.Level > self.Param['Seuil IM'] - (FirstPulse.Level - self.Param['Seuil IM'])
            if IMTreat:
                RejectIMThreshold = self.Param['Seuil sensi traitement'] + 2 * (FirstPulse.Level - self.Param['Seuil IM'])
        except:
            pass

        # This first index informs on the first pulse that can interact with other (annoying or visible pulses)
        InteractingId = len(Plateau.Pulses)

        # This second index informs on the first pulse that can be seen (only visible pulses)
        VisibleId = len(Plateau.Pulses)

        for pulse in list(reversed(Plateau.LevelPulses)):
            ThresholdVisibility = self.Param['Seuil sensi']
            ThresholdInteracting = self.Param['Seuil sensi'] - self.Param['Contraste geneur']
            if IMTreat:
                ThresholdVisibility = RejectIMThreshold

            if pulse.Level > ThresholdInteracting:
                InteractingId -= 1
            else:
                break

            if pulse.Level > ThresholdVisibility:
                VisibleId -= 1

        return InteractingId, VisibleId, Mono, HarmoTreat, RejectHThreshold

    # This function computes the interaction between pulses on a track with frequency Fe
    # Its outputs inform on the elementary detections on this track
    def TrackDetection(self, Plateau, Fe):
        if Plateau.IsEmpty():
            return []

        Ts = Plateau.StartingTime

        InteractingId, VisibleId, Mono, HarmoTreat, RejectHThreshold = self.TrackTreatment(Plateau)

        # We compute the frequency of the folded spectrum between 0 et Fe/2 for each interacting pulse and store the corresponding canal
        CanalList = []

        for Pulse in Plateau.LevelPulses[InteractingId:]:
            # We compute the frequency of the folded spectrum between 0 et Fe/2
            PulseFreq = Pulse.GetFreq(Ts)
            FoldFreq = min(PulseFreq - (PulseFreq // Fe) * Fe, -PulseFreq - (-PulseFreq // Fe) * Fe)

            # We compute the canal corresponding
            Canal = round(self.Param['Nint'] * FoldFreq / Fe)
            CanalList.append(Canal)

        # In case of harmonic processing
        if HarmoTreat:
            # We compute the folded frequency of the 2nd and 3rd harmonic of the brightest pulse
            # Those harmonics can mask other pulses
            PulseFreq = Plateau.LevelPulses[-1].GetFreq(Ts)
            FoldH2 = min(2 * PulseFreq - (2 * PulseFreq // Fe) * Fe, -2 * PulseFreq - (-2 * PulseFreq // Fe) * Fe)
            FoldH3 = min(3 * PulseFreq - (3 * PulseFreq // Fe) * Fe, -3 * PulseFreq - (-3 * PulseFreq // Fe) * Fe)

        # We built a list informing of which visible pulse is in fact masked or polluted
        MaskedList = []
        PollutedList = []

        for i in range(VisibleId, len(Plateau.Pulses)):
            VisiblePulse = Plateau.LevelPulses[i]
            Polluted = False
            Masked = False

            VisibleCanal = CanalList[i - InteractingId]

            # First and last canals are masked
            if VisibleCanal < self.Param['M1 aveugle'] or VisibleCanal > self.Param['Nint']/2 - self.Param['M2 aveugle']:
                PollutedList.append(True)
                MaskedList.append(True)
                continue

            # We check for the harmonic interaction
            if HarmoTreat:
                if abs(VisibleCanal - FoldH2) < 2 or abs(VisibleCanal - FoldH3) < 2:
                    if VisiblePulse.Level < RejectHThreshold:
                        PollutedList.append(True)
                        MaskedList.append(True)
                        continue

            # We check for interaction with other pulses by decreasing level
            for j in reversed(range(InteractingId, len(Plateau.Pulses))):
                InteractingPulse = Plateau.LevelPulses[j]
                InteractingCanal = CanalList[j - InteractingId]

                if InteractingPulse is VisiblePulse:
                    continue

                # if a pulse level is too low, the following pulse levels in the loop will be too low
                if VisiblePulse.Level - InteractingPulse.Level > self.Param['Contraste geneur 2']:
                    Masked = False
                    break

                else:
                    # if pulse canals are close, they will interact
                    if abs(VisibleCanal - InteractingCanal) < self.Param['M local']:
                        if VisiblePulse.Level < InteractingPulse.Level:
                            Polluted = True
                            Masked = True
                            break
                        else:
                            Polluted = True

            PollutedList.append(Polluted)
            MaskedList.append(Masked)

        # We make sure there is not more than "N DetEl" pulses transmitted
        # It has to be the strongest pulses in terms of level
        NDetEl = 0
        for i in reversed(range(len(MaskedList))):
            if not MaskedList[i]:
                if NDetEl == self.Param['N DetEl']:
                    MaskedList[i] = True
                    PollutedList[i] = True
                else:
                    NDetEl += 1

        return MaskedList, PollutedList

    # This function takes the elementary detections of each canal and fuse them into
    # NAD, simulating the behavior of the frequency ambiguity removal module
    def FreqAmbRemoval(self, Plateau, FeList):
        # We create scores which will determine whether a pulse is a NAD
        # Both of them must stay above 0 for a pulse to be a NAD
        ScoreMask = 1
        ScorePollution = 1
        for Fe in FeList:
            MaskedList, PollutedList = self.TrackDetection(Plateau, Fe)
            ScoreMask -= np.array(MaskedList) / 2
            ScorePollution -= np.array(PollutedList) / 2
        IsNAD = np.multiply(ScorePollution > 0 , ScoreMask > 0)

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
                if abs(Pulse.GetFreq(Ts) - meter.FreqCur) < self.Param['Seuil écart freq']:
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
            Tp = meter.Tc + self.Param['Durée maintien max']
            # If the meter did lose its pulse
            if Tp < Te:
                # We release it
                meter.Release(Tp)

                # We link the brightest NAD to the meter
                if len(LostNAD) > 0:
                    meter.Link(LostNAD.pop(0), Tp, Te)

            else:
                break
