from FakeDigitalTwin.XMLTools import loadXmlAsObj
from FakeDigitalTwinTranslator.Network import TransformerTranslator
import os
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt


local = r'C:\\Users\\matth\\OneDrive\\Documents\\Python\\Projets'
# local = r'C:\Users\Matthieu\Documents\Python\Projets'

EvaluationPulses = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'EvaluationPulsesAnt.xml'))
EvaluationPDWs = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'EvaluationPDWsDCI.xml'))

EvaluationSource = [[[Pulse['TOA'], Pulse['LI'], Pulse['Level'], Pulse['FreqStart'], Pulse['FreqEnd']] for Pulse in PulsesAnt] for PulsesAnt in EvaluationPulses]
EvaluationTranslation = [[[PDW['TOA'], PDW['LI'], PDW['Level'], PDW['FreqMin'], PDW['FreqMax'], int('CW' in PDW['flags']), int('TroncAv' in PDW['flags']),
                         int(len(PDW['flags']) == 0)] for PDW in PDWsDCI] for PDWsDCI in EvaluationPDWs]

d_source = 5
d_target = 5
d_input_Enc = 32
d_input_Dec = 32
d_att = 32
num_flags = 3
num_heads = 4
len_target = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Translator = TransformerTranslator(d_source=d_source, d_target=d_target, d_att=d_att, d_input_Enc=d_input_Enc, target_len=len_target,
                                   d_input_Dec=d_input_Dec, num_flags=num_flags, num_heads=num_heads, device=device)

Translator.load_state_dict(torch.load(os.path.join(local, 'FakeDigitalTwinTranslator', 'Translator')))
Translator.eval()


EvaluationSource = torch.tensor(EvaluationSource)
EvaluationTranslation = pad_sequence([torch.tensor(el) for el in EvaluationTranslation], batch_first=True)
_, temp_len, _ = EvaluationTranslation.shape
EvaluationTranslation = F.pad(EvaluationTranslation, (0, 0, 0, len_target-temp_len))
EvaluationEnded = (torch.norm(EvaluationTranslation, dim=-1) == 0).unsqueeze(-1).to(torch.float32)

evaluation_size = len(EvaluationSource)

EvaluationSource = EvaluationSource.to(device=device, dtype=torch.float32)
EvaluationTranslation = EvaluationTranslation.to(device=device, dtype=torch.float32)
EvaluationEnded = EvaluationEnded.to(device=device, dtype=torch.float32)

# NumWords = torch.sum((1 - EvaluationEnded), dim=1).unsqueeze(-1).to(torch.int32)
# ActionMask = (1 - EvaluationEnded) / NumWords
# for i in range(len(NumWords)):
#     ActionMask[(i, int(NumWords[i]))] = 1
#
# with torch.no_grad():
#     PredictedTranslation, PredictedAction = Translator.forward(source=EvaluationSource, target=EvaluationTranslation)
#
# PredictedTranslation = PredictedTranslation[:, 1:]
# PredictedAction = PredictedAction[:, :-1]
# err = (torch.norm((EvaluationTranslation - PredictedTranslation) * (1 - EvaluationEnded)) +
#        torch.norm((PredictedAction - (1 - EvaluationEnded)) * ActionMask)) / evaluation_size


Translation = torch.zeros(size=EvaluationTranslation.shape).to(device)
# Translation.shape = (batch_size, len_target, d_target + num_flags)
batch_size = len(EvaluationTranslation)

with torch.no_grad():
    for i in tqdm(range(len_target)):
        if i == len_target-1:
            Translation, Actions = Translator.forward(source=EvaluationSource, target=Translation)
        else:
            Translation, _ = Translator.forward(source=EvaluationSource, target=Translation)
        Translation = Translation[:, 1:]
        Translation[:, len_target+1:] = 0

Actions = Actions[:, 1:, 0]
MaskEndTranslation = torch.zeros(size=EvaluationTranslation.shape)
for i in range(batch_size):
    for j in range(len_target):
        if Actions[i, j] > 0.5:
            MaskEndTranslation[i, j] = 1
        else:
            break



