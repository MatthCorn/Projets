from FakeDigitalTwin.XMLTools import loadXmlAsObj
from FakeDigitalTwinTranslator.ClassicNetwork import TransformerTranslator
import os
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Ce script sert à, l'apprentissage du réseau Network.TransformerTranslator

local = r'C:\\Users\\matth\\OneDrive\\Documents\\Python\\Projets'
# local = r'C:\Users\Matthieu\Documents\Python\Projets'
TrainingPulses = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'TrainingPulsesAnt.xml'))
TrainingPDWs = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'TrainingPDWsDCI.xml'))
ValidationPulses = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'ValidationPulsesAnt.xml'))
ValidationPDWs = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'ValidationPDWsDCI.xml'))


TrainingSource = [[[Pulse['TOA'], Pulse['LI'], Pulse['Level'], Pulse['FreqStart'], Pulse['FreqEnd']] for Pulse in PulsesAnt] for PulsesAnt in TrainingPulses]
TrainingTranslation = [[[PDW['TOA'], PDW['LI'], PDW['Level'], PDW['FreqMin'], PDW['FreqMax'], int('CW' in PDW['flags']), int('TroncAv' in PDW['flags']),
                         int(len(PDW['flags']) == 0)] for PDW in PDWsDCI] for PDWsDCI in TrainingPDWs]

ValidationSource = [[[Pulse['TOA'], Pulse['LI'], Pulse['Level'], Pulse['FreqStart'], Pulse['FreqEnd']] for Pulse in PulsesAnt] for PulsesAnt in ValidationPulses]
ValidationTranslation = [[[PDW['TOA'], PDW['LI'], PDW['Level'], PDW['FreqMin'], PDW['FreqMax'], int('CW' in PDW['flags']), int('TroncAv' in PDW['flags']),
                         int(len(PDW['flags']) == 0)] for PDW in PDWsDCI] for PDWsDCI in ValidationPDWs]

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
TrainingSource = torch.tensor(TrainingSource)
# Source.shape = (batch_size, len_input, d_input)

# Rajoute des 0 à la fin des scénarios de PDWs pour qu'ils aient toutes la même longueure
TrainingTranslation = pad_sequence([torch.tensor(el) for el in TrainingTranslation], batch_first=True)

_, temp_len, _ = TrainingTranslation.shape
TrainingTranslation = F.pad(TrainingTranslation, (0, 0, 0, len_target-temp_len))
# Translation.shape = (batch_size, len_target, d_target + num_flags)


# Même travail sur l'ensemble de validation
ValidationSource = torch.tensor(ValidationSource)
TrainingEnded = (torch.norm(TrainingTranslation, dim=-1) == 0).unsqueeze(-1).to(torch.float32)
ValidationTranslation = pad_sequence([torch.tensor(el) for el in ValidationTranslation], batch_first=True)
_, temp_len, _ = ValidationTranslation.shape
ValidationTranslation = F.pad(ValidationTranslation, (0, 0, 0, len_target-temp_len))
ValidationEnded = (torch.norm(ValidationTranslation, dim=-1) == 0).unsqueeze(-1).to(torch.float32)

batch_size = 50
training_size = len(TrainingSource)
validation_size = len(ValidationSource)

# Procédure d'entrainement
optimizer = torch.optim.Adam(Translator.parameters(), weight_decay=1e-5, lr=3e-5)
TrainingErrList = []
ValidationErrList = []
for i in tqdm(range(200)):
    n_batch = int(training_size/batch_size)
    TrainingError = 0
    for j in range(n_batch):
        optimizer.zero_grad(set_to_none=True)

        BatchSource = TrainingSource[j*batch_size: (j+1)*batch_size].detach().to(device=device, dtype=torch.float32)
        BatchTranslation = TrainingTranslation[j*batch_size: (j+1)*batch_size].detach().to(device=device, dtype=torch.float32)
        BatchEnded = TrainingEnded[j*batch_size: (j+1)*batch_size].detach().to(device=device, dtype=torch.float32)

        # Pour faire en sorte que l'action "continuer d'écrire" soit autant représentée en True qu'en False, on pondère le masque des actions
        NumWords = torch.sum(BatchEnded, dim=1).unsqueeze(-1).to(torch.int32)
        BatchActionMask = (1 - BatchEnded) / NumWords
        for i in range(len(NumWords)):
            BatchActionMask[(i, int(NumWords[i]))] = 1

        PredictedTranslation, PredictedAction = Translator.forward(source=BatchSource, target=BatchTranslation)

        # Comme la décision d'arrêter d'écrire passe par Action, et pas par un token <end>, on ne s'interesse pas au dernier mot
        # On ne s'interesse pas non plus à l'action prédite par le token <start>, puisque de toute manière il est suivi d'un mot
        PredictedTranslation = PredictedTranslation[:, 1:]
        PredictedAction = PredictedAction[:, :-1]
        err = (torch.norm((BatchTranslation - PredictedTranslation)*(1-BatchEnded)) + torch.norm((PredictedAction - (1-BatchEnded))*BatchActionMask))/batch_size
        err.backward()
        optimizer.step()
        TrainingError += float(err)*batch_size
    TrainingErrList.append(TrainingError/training_size)

    n_batch = int(validation_size / batch_size)
    ValidationError = 0
    for j in range(n_batch):

        BatchSource = ValidationSource[j * batch_size: (j + 1) * batch_size].detach().to(device=device, dtype=torch.float32)
        BatchTranslation = ValidationTranslation[j * batch_size: (j + 1) * batch_size].detach().to(device=device, dtype=torch.float32)
        BatchEnded = ValidationEnded[j * batch_size: (j + 1) * batch_size].detach().to(device=device, dtype=torch.float32)

        # On ajoute un zero au début pour décaler le masque vers la droite et voir si le réseau prédit correctement l'action d'arrêter l'écriture
        BatchActionMask = (1 - F.pad(BatchEnded, (0, 0, 1, 0))[:, :len_target])

        with torch.no_grad():
            PredictedTranslation, PredictedAction = Translator.forward(source=BatchSource, target=BatchTranslation)

        # Comme la décision d'arrêter d'écrire passe par Action, et pas par un token <end>, on ne s'interesse pas au dernier mot
        # On ne s'interesse pas non plus à l'action prédite par le token <start>, puisque de toute manière il est suivi d'un mot
        PredictedTranslation = PredictedTranslation[:, 1:]
        PredictedAction = PredictedAction[:, :-1]
        err = (torch.norm((BatchTranslation - PredictedTranslation) * (1 - BatchEnded)) + torch.norm((PredictedAction - (1 - BatchEnded)) * BatchActionMask)) / batch_size
        ValidationError += float(err)*batch_size
    ValidationErrList.append(ValidationError/validation_size)

plt.plot(TrainingErrList, 'b')
plt.plot(ValidationErrList, 'r')
plt.show()

