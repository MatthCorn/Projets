from Tools.XMLTools import loadXmlAsObj
import os
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import shutil

local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'OneDrive', 'Documents', 'Python', 'Projets')
# local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'Documents', 'Python', 'Projets')


# Temps de maintien max d'un mesureur sans voir son impulsion
HoldingTime = 0.5
NbMaxPulses = 10
RangeAvg = 1
DeltaT = 0.5
BatchSize = 10000000
TMax = 35

# Cette fonction donne un majorant de la date de publication d'un PDW donné
def TimeRelease(PDW):
    # PDW est une liste de la forme [TOA: float, LI: float, Level: float, FreqMin: float, FreqMax: float, flag1: int, flag2: int, flag3: int]
    Maj = PDW[0] + PDW[1] + HoldingTime
    return Maj

# Cette fonction a pour rôle de transformer les phrases de Source et Translation en plein de petites
# phrases correspondant au découpage par salve temporelle de longueur DeltaT
def Spliter(Source, DeltaT):
    NewSource = []
    NewTranslation = []
    batch_len = len(Source)
    for i in range(batch_len):
        SourceSentence = Source[i]

        # On commence par découper la phrase de Source sur les intervalles de taille DeltaT
        SplitedSourceSentence = []

        # Ces indices permettent de suivre les impulsions déjà ajoutées (comme elles sont triées par TOA)
        SourceId = 0
        t = DeltaT
        while not SourceId == len(SourceSentence):
            SourceBursts = []
            while (SourceId < len(SourceSentence)) and (SourceSentence[SourceId][0] < t):
                SourceBursts.append(SourceSentence[SourceId])
                SourceId += 1

            t += DeltaT

            SplitedSourceSentence.append(SourceBursts)

        # On reconstruit les phrases telles qu'elles seront données au traducteur
        Remain = [[0] * 5] * NbMaxPulses
        for Bursts in SplitedSourceSentence:
            Remain = (Remain + Bursts)[-NbMaxPulses:]
            if Remain[0] != [[0] * 5]:
                NewSource.append(Remain)

                TrRemain = [max(Remain[i-1], Remain[i], Remain[i+1], key=lambda x:x[2]) for i in range(RangeAvg, len(Remain)-RangeAvg)]
                NewTranslation.append(TrRemain)

    return NewSource, NewTranslation

def FDTDataMaker():
    for TypeData in ['Validation', 'Training']:
        Pulses = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', TypeData + 'PulsesAnt.xml'))

        Source = [[[Pulse['TOA'], Pulse['LI'], Pulse['Level'], Pulse['FreqStart'], Pulse['FreqEnd']]
                   for Pulse in PulsesAnt]
                  for PulsesAnt in Pulses]

        # Ici les données sont founies en batch
        # Source est une liste de séquence d'impulsions de même longueur

        Source, Translation = Spliter(Source, DeltaT)

        # On transforme ces listes en tenseur
        Source = torch.tensor(Source)
        # Source.shape = (batch_size, NbMaxPulses, d_input)

        # Rajoute des 0 à la fin des scénarios de PDWs pour qu'ils aient toutes la même longueure
        Translation = pad_sequence([torch.tensor(el) for el in Translation], batch_first=True)

        # Translation.shape = (batch_size, NbMaxPulses, d_input = d_output)

        WriteBatchs(Source=Source, Translation=Translation, BatchSize=BatchSize, TypeData=TypeData)


def WriteBatchs(Source, Translation, BatchSize, TypeData):
    save_path = os.path.join(local, 'StepByStep', 'S1', 'BurstsData', 'Data')
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