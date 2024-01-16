from Tools.XMLTools import loadXmlAsObj
import os
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import shutil

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')


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
                if SourceSentence[SourceId][0] > t:
                    break
                SourceBursts.append(SourceSentence[SourceId])
                SourceId += 1

            # On ajoute l'impulsion informant de la date de fin de la salve
            SourceBursts.append([t, 0, 0, 0, 0])
            SplitedSourceSentence.append(SourceBursts)

            TranslationBursts = []
            while not TranslationId == len(TranslationSentence):
                if TimeRelease(TranslationSentence[TranslationId]) > t:
                    break
                TranslationBursts.append(TranslationSentence[TranslationId])
                TranslationId += 1
            SplitedTranslationSentence.append(TranslationBursts)

        while t < TMax:
            t += DeltaT
            SplitedSourceSentence.append([[t, 0, 0, 0, 0]])
            SplitedTranslationSentence.append([])

        if Eval:
            # On reconstruit la liste les salves de mots composants la phrase
            Sentence = []
            Remain = [[0] * 5] * (NbMaxPulses + 1)
            for Bursts in SplitedSourceSentence:
                Remain = (Remain[:-1] + Bursts)[-(NbMaxPulses + 1):]
                Sentence.append(Remain)
            NewSource.append(Sentence)

        else:
            # On reconstruit les phrases telles qu'elles seront données au traducteur
            Remain = [[0] * 5] * (NbMaxPulses + 1)
            for Bursts in SplitedSourceSentence:
                Remain = (Remain[:-1] + Bursts)[-(NbMaxPulses + 1):]
                NewSource.append(Remain)

        Remain = [[0] * 8] * NbPDWsMemory
        for Bursts in SplitedTranslationSentence:
            Remain = Remain[-NbPDWsMemory:] + Bursts
            NewTranslation.append(Remain)

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
    save_path = os.path.join(local, 'FDTTranslator', 'Bursts', 'FPIC', 'BurstsData', 'Data')
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