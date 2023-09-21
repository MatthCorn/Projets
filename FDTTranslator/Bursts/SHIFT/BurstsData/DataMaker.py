from Tools.XMLTools import loadXmlAsObj
import os
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import shutil

local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'OneDrive', 'Documents', 'Python', 'Projets')
# local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'Documents', 'Python', 'Projets')

# Temps de maintien max d'un mesureur sans voir son impulsion
HoldingTime = 0.5
NbMaxPulses = 10
NbPDWsMemory = 10
DeltaT = 0.5
NbMaxPDWs = 20
BatchSize = 10000000
TMax = 35

# Cette fonction donne un majorant de la date de publication d'un PDW donné
def TimeRelease(PDW):
    # PDW est une liste de la forme [TOA: float, LI: float, Level: float, FreqMin: float, FreqMax: float, flag1: int, flag2: int, flag3: int]
    Maj = PDW[0] + PDW[1] + HoldingTime
    return Maj

# Cette fonction a pour rôle de transformer les phrases de Source et Translation en plein de petites
# phrases correspondant au découpage par salve temporelle de longueur DeltaT
def Spliter(Source, Translation, DeltaT, Eval=False):
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
                Pulse = SourceSentence[SourceId]
                if Pulse[0] > t:
                    break
                Pulse[0] -= t
                SourceBursts.append(Pulse)
                SourceId += 1
            SplitedSourceSentence.append(SourceBursts)

            TranslationBursts = []
            while not TranslationId == len(TranslationSentence):
                PDW = TranslationSentence[TranslationId]
                if PDW[0] > t:
                    break
                PDW[0] -= t
                TranslationBursts.append(PDW)
                TranslationId += 1
            SplitedTranslationSentence.append(TranslationBursts)

        while t < TMax:
            t += DeltaT
            SplitedSourceSentence.append([])
            SplitedTranslationSentence.append([])

        if Eval:
            # On reconstruit la liste les salves de mots composants la phrase
            Sentence = []
            Remain = [[0] * 5] * NbMaxPulses
            for Bursts in SplitedSourceSentence:
                Remain = (Remain + Bursts)[-NbMaxPulses:]
                Sentence.append(Remain)
            NewSource.append(Sentence)

        else:
            # On reconstruit les phrases telles qu'elles seront données au traducteur
            Remain = [[0] * 5] * NbMaxPulses
            t = 0
            for Bursts in SplitedSourceSentence:
                t += DeltaT
                Remain = (Remain + Bursts)[-NbMaxPulses:]
                NewSource.append((np.array(Remain) - [t, 0, 0, 0, 0]).tolist())

        Remain = [[0] * 8] * NbPDWsMemory
        t = 0
        for Bursts in SplitedTranslationSentence:
            Remain = Remain[-NbPDWsMemory:] + Bursts
            NewTranslation.append((np.array(Remain) - [t, 0, 0, 0, 0, 0, 0, 0]).tolist())

    if Eval:
        return NewSource, Translation

    return NewSource, NewTranslation

def FDTDataMaker():
    for TypeData in ['Validation', 'Training', 'Evaluation']:
        Pulses = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', TypeData + 'PulsesAnt.xml'))
        PDWs = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', TypeData + 'PDWsDCI.xml'))

        Source = [
            [[Pulse['TOA'], Pulse['LI'], Pulse['Level'], Pulse['FreqStart'], Pulse['FreqEnd']] for Pulse in PulsesAnt]
            for PulsesAnt in Pulses]
        Translation = [[[PDW['TOA'], PDW['LI'], PDW['Level'], PDW['FreqMin'], PDW['FreqMax'], int('CW' in PDW['flags']),
                         int('TroncAv' in PDW['flags']),
                         int(len(PDW['flags']) == 0)] for PDW in PDWsDCI] for PDWsDCI in PDWs]

        # Ici les données sont founies en batch
        # Source est une liste de séquence d'impulsions de même longueur
        # Translation est une liste de séquence de PDWs, elles peuvent être de longueures différentes


        Eval = (TypeData == 'Evaluation')
        Source, Translation = Spliter(Source, Translation, DeltaT, Eval=Eval)
        # On transforme ces listes en tenseur
        Source = torch.tensor(Source)
        # Source.shape = (batch_size, len_input, d_input)

        # Rajoute des 0 à la fin des scénarios de PDWs pour qu'ils aient toutes la même longueure
        Translation = pad_sequence([torch.tensor(el) for el in Translation], batch_first=True)

        if not Eval:
            _, temp_len, _ = Translation.shape
            Translation = F.pad(Translation, (0, 0, 0, NbMaxPDWs - temp_len))
            # Translation.shape = (batch_size, NbMaxPDWs, d_target + num_flags)

        WriteBatchs(Source=Source, Translation=Translation, BatchSize=BatchSize, TypeData=TypeData)


def WriteBatchs(Source, Translation, BatchSize, TypeData):
    save_path = os.path.join(local, 'FDTTranslator', 'Bursts', 'SHIFT', 'BurstsData', 'Data')
    try:
        shutil.rmtree(os.path.join(save_path, TypeData))
    except:
        None

    os.mkdir(os.path.join(save_path, TypeData))

    NbBatch, r = divmod(len(Source), BatchSize)
    NbBatch += (r != 0)
    for i in range(NbBatch):
        BatchSource = Source[i*BatchSize:(i+1)*BatchSize].numpy()
        BatchTranslation = Translation[i*BatchSize:(i+1)*BatchSize].numpy()
        SourceFName = os.path.join(save_path, TypeData, 'PulsesAnt_{}.npy'.format(i))
        TranslationFName = os.path.join(save_path, TypeData, 'PDWsDCI_{}.npy'.format(i))
        np.save(SourceFName, BatchSource)
        np.save(TranslationFName, BatchTranslation)

if __name__ == '__main__':
    FDTDataMaker()