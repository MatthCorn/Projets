from FakeDigitalTwin.XMLTools import loadXmlAsObj
from FakeDigitalTwinTranslator.Network import TransformerTranslator
import os
import torch
from tqdm import tqdm

local = r'C:\\Users\\matth\\OneDrive\\Documents\\Python\\Projets'
# local = r'C:\Users\Matthieu\Documents\Python\Projets'
BatchPulses = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'PulsesAnt.xml'))
BatchPDWs = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'PDWsDCI.xml'))

BatchPulses = [[[Pulse['TOA'], Pulse['LI'], Pulse['Level'], Pulse['FreqStart'], Pulse['FreqEnd']] for Pulse in PulsesAnt] for PulsesAnt in BatchPulses]
BatchPDWs = [[[PDW['TOA'], PDW['LI'], PDW['Level'], PDW['FreqMin'], PDW['FreqMax'], int('CW' in PDW['flags']), int('TroncAv' in PDW['flags']),
               int(len(PDW['flags']) == 0)] for PDW in PDWsDCI] for PDWsDCI in BatchPDWs]

d_source = 5
d_target = 5
d_input_Enc = 32
d_input_Dec = 32
d_att = 32
num_flags = 3
num_heads = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

Translator = TransformerTranslator(d_source=d_source, d_target=d_target, d_att=d_att, d_input_Enc=d_input_Enc,
                                   d_input_Dec=d_input_Dec, num_flags=num_flags, num_heads=num_heads, device=device)

if False:
    from torch.nn.utils.rnn import pad_sequence

    source = BatchPulses
    translatedSource = BatchPDWs
    # Ici les données sont founies en batch
    # source est une liste de séquence de même taille
    source_input = torch.tensor(source)

    # translatedSource est une liste de traduction, elles peuvent être de tailles différentes
    LenList = [len(el) for el in translatedSource]
    len_target = max(LenList)

    translatedSource = pad_sequence([torch.tensor(sublist) for sublist in translatedSource], batch_first=True)
    # translatedSource.shape = (batch_size, len_target, d_target)

    Error = 0
    for i in range(len_target):
        print(i)
        # On donne à chaque fois la source et les "i" premiers mots de la traduction et on compare le mot prédit
        ended = (torch.tensor(LenList) <= i).unsqueeze(-1).unsqueeze(-1).to(dtype=torch.float32)
        # ended.shape = (batch_size, 1, 1)
        target_input = translatedSource[:, :i]
        expected_prediction = translatedSource[:, i]
        actual_prediction, action = Translator.forward(source=source_input, target=target_input, ended=ended)

        Error += torch.norm(expected_prediction - actual_prediction) + torch.norm(action - ended)


optimizer = torch.optim.Adam(Translator.parameters(), weight_decay=1e-6, lr=3e-4)
ErrList = []
for i in tqdm(range(100)):
    err = Translator.Error(source=BatchPulses, translatedSource=BatchPDWs)
    err.backward()
    optimizer.step()
    ErrList.append(float(err))
