from Tools.XMLTools import loadXmlAsObj
import os
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def FDTDataLoader(ListTypeData=[], len_target=100, local='', variables_dict={}):

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
