from FakeDigitalTwin.XMLTools import loadXmlAsObj
from FakeDigitalTwinTranslator.Network import TransformerTranslator
import os
import torch
import tqdm

# local = r'C:\\Users\\matth\\OneDrive\\Documents\\Python\\Projets'
local = r'C:\Users\Matthieu\Documents\Python\Projets'
BatchPulses = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'PulsesAnt.xml'))
BatchPDWs = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'PDWsDCI.xml'))

BatchPulses = [[[Pulse['TOA'], Pulse['LI'], Pulse['Level'], Pulse['FreqStart'], Pulse['FreqEnd']] for Pulse in PulsesAnt] for PulsesAnt in BatchPulses]
BatchPDWs = [[[PDW['TOA'], PDW['LI'], PDW['Level'], PDW['FreqMin'], PDW['FreqMax'], int('CW' in PDW['flags']), int('TroncAv' in PDW['flags']),
               int(len(PDW['flags']) == 0)] for PDW in PDWsDCI] for PDWsDCI in BatchPDWs]

d_source = 5
d_target = 8
d_input_Enc = 32
d_input_Dec = 32
d_att = 32
num_heads = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Translator = TransformerTranslator(d_source=d_source, d_target=d_target, d_input_Enc=d_input_Enc, d_input_Dec=d_input_Dec, num_heads=num_heads).to(device)

optimizer = torch.optim.Adam(Translator.parameters(), weight_decay=1e-6, lr=3e-4)
ErrList = []
for i in tqdm(range(100)):
    err = Translator.Error(source=BatchPulses, translatedSource=BatchPDWs)
    err.backward()
    optimizer.step()
    ErrList.append(float(err))
