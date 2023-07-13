from FakeDigitalTwin.XMLTools import loadXmlAsObj
from FakeDigitalTwinTranslator.Network import TransformerTranslator
import os
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import random
import matplotlib.pyplot as plt

# local = r'C:\\Users\\matth\\OneDrive\\Documents\\Python\\Projets'
local = r'C:\Users\Matthieu\Documents\Python\Projets'
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

Translator = TransformerTranslator(d_source=d_source, d_target=d_target, d_att=d_att, d_input_Enc=d_input_Enc, target_len=100,
                                   d_input_Dec=d_input_Dec, num_flags=num_flags, num_heads=num_heads, device=device)


# Pour l'entrainement, il nous faut en fait des listes de tuples (impulsions antenne - début des PDWs - prochain PDW),
# il faut  donc transformer  BatchPDWs pour créer à partir de chaque liste de PDWs tous les couples (début des PDWs - prochain PDW)

# Ici les données sont founies en batch
# BatchPulses est une liste de séquence de même taille

# BatchPDWs est une liste de traduction, elles peuvent être de tailles différentes
# On crée une liste de début de séquence de PDWs et une liste de la PDW suivante à prédire
# La liste de l'action à prédire correspond si on arrête ou pas la génération (on a fini de traduire les Pulses en PDWs)
LenBatch = len(BatchPDWs)
# LenBatch = len(BatchPulses) aussi
Source = []
PartialTranslation = []
NextPDW = []
NextAction = []
for i in range(LenBatch):
    PDWsList = BatchPDWs[i]
    for j in range(len(PDWsList)):
        Source.append(BatchPulses[i])
        NextPDW.append(PDWsList[j])
        NextAction.append(0)

        # On fait directement cela pour le besoin future du passage en tensor
        PartialTranslation.append(torch.tensor(PDWsList[:j]).reshape(-1, d_target+num_flags))

    Source.append(BatchPulses[i])
    NextPDW.append([0]*(d_target+num_flags))
    NextAction.append(1)
    PartialTranslation.append(torch.tensor(PDWsList))


# On transforme ces listes en tenseur
Source = torch.tensor(Source)
# Source.shape = (batch_size, len_input, d_input)
data_size = len(Source)
NextPDW = torch.tensor(NextPDW)
# NextPDW.shape = (batch_size, d_target + num_flags)
NextAction = torch.tensor(NextAction).unsqueeze(-1).unsqueeze(-1)
# NextAction.shape = (batch_size, 1, 1)

# Rajoute des 0 à la fin des traductions partielles pour qu'elles aient toutes la même taille
PartialTranslation = pad_sequence(PartialTranslation, batch_first=True)
# PartialTranslation.shape = (batch_size, len_target, d_target + num_flags)

batch_size = 50

# On mélange les entrées pour éviter des biais d'apprentissage (car on va découper nos données en batchs)
random.seed(0)
IdList = list(range(data_size))
random.shuffle(IdList)
Source, NextPDW, NextAction, PartialTranslation = Source[IdList], NextPDW[IdList], NextAction[IdList], PartialTranslation[IdList]

# Procédure d'entrainement
optimizer = torch.optim.Adam(Translator.parameters(), weight_decay=1e-5, lr=3e-5)
ErrList = []
for i in tqdm(range(2000)):
    n_batch = int(data_size/batch_size)
    for j in range(n_batch):
        print(j)
        optimizer.zero_grad(set_to_none=True)

        BatchSource = Source[j*n_batch: (j+1)*n_batch].detach().to(device=device, dtype=torch.float32)
        BatchPartialTranslation = PartialTranslation[j*n_batch: (j+1)*n_batch].detach().to(device=device, dtype=torch.float32)
        BatchNextPDW = NextPDW[j*n_batch: (j+1)*n_batch].detach().to(device=device, dtype=torch.float32)
        BatchNextAction = NextAction[j*n_batch: (j+1)*n_batch].detach().to(device=device, dtype=torch.float32)

        PredictedPDW, PredictedAction = Translator.forward(source=BatchSource, target=BatchPartialTranslation, ended=BatchNextAction)
        err = torch.norm(BatchNextPDW - PredictedPDW) + torch.norm(BatchNextAction - PredictedAction)
        err.backward()
        optimizer.step()
        ErrList.append(float(err))

plt.plot(ErrList)
plt.show()

