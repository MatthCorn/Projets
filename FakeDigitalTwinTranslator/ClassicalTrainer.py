from FakeDigitalTwin.XMLTools import loadXmlAsObj
from FakeDigitalTwinTranslator.ClassicNetwork import TransformerTranslator
import os
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Ce script sert à, l'apprentissage du réseau Network.TransformerTranslator

# local = r'C:\\Users\\matth\\OneDrive\\Documents\\Python\\Projets'
local = r'C:\Users\Matthieu\Documents\Python\Projets'
BatchPulses = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'PulsesAnt.xml'))
BatchPDWs = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'PDWsDCI.xml'))

Source = [[[Pulse['TOA'], Pulse['LI'], Pulse['Level'], Pulse['FreqStart'], Pulse['FreqEnd']] for Pulse in PulsesAnt] for PulsesAnt in BatchPulses]
Translation = [[[PDW['TOA'], PDW['LI'], PDW['Level'], PDW['FreqMin'], PDW['FreqMax'], int('CW' in PDW['flags']), int('TroncAv' in PDW['flags']),
               int(len(PDW['flags']) == 0)] for PDW in PDWsDCI] for PDWsDCI in BatchPDWs]

d_source = 5
d_target = 5
d_input_Enc = 32
d_input_Dec = 32
d_att = 32
num_flags = 3
num_heads = 4
len_target = 150

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Translator = TransformerTranslator(d_source=d_source, d_target=d_target, d_att=d_att, d_input_Enc=d_input_Enc, target_len=len_target,
                                   d_input_Dec=d_input_Dec, num_flags=num_flags, num_heads=num_heads, device=device)


# Ici les données sont founies en batch
# Source est une liste de séquence d'impulsions de même longueur
# Translation est une liste de séquence de PDWs, elles peuvent être de longueures différentes

# On transforme ces listes en tenseur
Source = torch.tensor(Source)
# Source.shape = (batch_size, len_input, d_input)

# Rajoute des 0 à la fin des scénarios de PDWs pour qu'ils aient toutes la même longueure
Translation = pad_sequence([torch.tensor(el) for el in Translation], batch_first=True)
_, temp_len, _ = Translation.shape
Translation = F.pad(Translation, (0, 0, 0, len_target-temp_len))
# Translation.shape = (batch_size, len_target, d_target + num_flags)

Ended = (Translation == 0).to(torch.float32)

batch_size = 100
data_size = len(Source)

# Procédure d'entrainement
optimizer = torch.optim.Adam(Translator.parameters(), weight_decay=1e-5, lr=3e-5)
ErrList = []
for i in tqdm(range(100)):
    n_batch = int(data_size/batch_size)
    for j in range(n_batch):
        optimizer.zero_grad(set_to_none=True)

        BatchSource = Source[j*batch_size: (j+1)*batch_size].detach().to(device=device, dtype=torch.float32)
        BatchTranslation = Translation[j*batch_size: (j+1)*batch_size].detach().to(device=device, dtype=torch.float32)
        BatchEnded = Ended[j*batch_size: (j+1)*batch_size].detach().to(device=device, dtype=torch.float32)

        # On ajoute un zero au début pour décaler le masque vers la droite et voir si le réseau prédit correctement l'action d'arrêter l'écriture
        BatchActionMask = (1 - F.pad(BatchEnded, (0, 0, 1, 0))[:, len_target])

        PredictedTranslation, PredictedAction = Translator.forward(source=BatchSource, target=BatchTranslation)
        err = torch.norm((BatchTranslation - PredictedTranslation)*(1-BatchEnded)) + torch.norm((PredictedAction - (1-BatchEnded))*BatchActionMask)
        err.backward()
        optimizer.step()
        ErrList.append(float(err))

plt.plot(ErrList)
plt.show()

