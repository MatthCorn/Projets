from FakeDigitalTwin.XMLTools import loadXmlAsObj
import os
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# Temps de maintien max d'un mesureur sans voir son impulsion
HoldingTime = 0.5
NbMaxPulsesTranslator = 10
NbPDWsMemory = 5
DeltaT = 0.2
def FDTDataLoader(ListTypeData=[], len_target=15, local='', variables_dict={}):

    NewArg = []
    NewValue = []
    for TypeData in ['Training', 'Validation', 'Evaluation']:
        if TypeData in ListTypeData:
            Pulses = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', TypeData + 'PulsesAnt.xml'))
            PDWs = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', TypeData + 'PDWsDCI.xml'))

            Source = [[[Pulse['TOA'], Pulse['LI'], Pulse['Level'], Pulse['FreqStart'], Pulse['FreqEnd']] for Pulse in PulsesAnt] for PulsesAnt in Pulses]
            Translation = [[[PDW['TOA'], PDW['LI'], PDW['Level'], PDW['FreqMin'], PDW['FreqMax'], int('CW' in PDW['flags']), int('TroncAv' in PDW['flags']),
                             int(len(PDW['flags']) == 0)] for PDW in PDWsDCI] for PDWsDCI in PDWs]

            # Ici les données sont founies en batch
            # Source est une liste de séquence d'impulsions de même longueur
            # Translation est une liste de séquence de PDWs, elles peuvent être de longueures différentes

            Source, Translation = Spliter(Source, Translation, DeltaT)
            # On transforme ces listes en tenseur
            Source = torch.tensor(Source)
            # Source.shape = (batch_size, len_input, d_input)

            # Rajoute des 0 à la fin des scénarios de PDWs pour qu'ils aient toutes la même longueure
            Translation = pad_sequence([torch.tensor(el) for el in Translation], batch_first=True)

            _, temp_len, _ = Translation.shape
            Translation = F.pad(Translation, (0, 0, 0, len_target - temp_len))
            # Translation.shape = (batch_size, len_target, d_target + num_flags)

            NewArg.append(TypeData + 'Source')
            NewValue.append(Source)
            NewArg.append(TypeData + 'Translation')
            NewValue.append(Translation)

    for i in range(len(NewArg)):
        variables_dict.__setitem__(NewArg[i], NewValue[i])

def LoadParam(dict, variables_dict):
    for key in dict.keys():
        variables_dict.__setitem__(key, dict[key])


# Cette fonction donne un majorant de la date de publication d'un PDW donné
def TimeRelease(PDW):
    # PDW est une liste de la forme [TOA: float, LI: float, Level: float, FreqMin: float, FreqMax: float, flag1: int, flag2: int, flag3: int]
    Maj = PDW[0] + PDW[1] + HoldingTime
    return Maj


def Spliter(Source, Translation, DeltaT):
    NewSource = []
    NewTranslation = []
    batch_len = len(Source)
    for i in range(batch_len):
        SourceSentence = Source[i]
        TranslationSentence = Translation[i]
        # Il faut trier les PDWs par date de publication
        TranslationSentence.sort(key=TimeRelease)

        # On commence par découper la phrase de Source et la phrase de Translation sur les intervalles de taille DeltaT
        SplitedSourceSentence = []
        SplitedTranslationSentence = []

        # Ces indices permettent de suivre les impulsions déjà ajoutées (comme elles sont triées par TOA)
        SourceId = 0
        TranslationId = 0
        t = 0
        while not (SourceId == len(SourceSentence) and TranslationId == len(TranslationSentence)):
            t += DeltaT
            SourceBursts = []
            while not SourceId == len(SourceSentence):
                if SourceSentence[SourceId][SourceId] > t:
                    break
                SourceBursts.append(SourceSentence[SourceId])
                SourceId += 1
            SplitedSourceSentence.append(SourceBursts)

            TranslationBursts = []
            while not TranslationId == len(TranslationSentence):
                if TranslationSentence[TranslationId][TranslationId] > t:
                    break
                TranslationBursts.append(TranslationSentence[TranslationId])
                TranslationId += 1
            SplitedTranslationSentence.append(TranslationBursts)

        # On reconstruit les phrases telles qu'elles seront données au traducteur
        Remain = [[0]*5]*NbMaxPulsesTranslator
        for Bursts in SourceBursts:
            Remain = (Remain + Bursts)[-NbMaxPulsesTranslator:]
            NewSource.append(Remain)

        Remain = [[0]*8]*NbPDWsMemory
        for Bursts in TranslationBursts:
            Remain = Remain[-NbPDWsMemory:] + Bursts
            NewTranslation.append(Remain)
    return NewSource, NewTranslation