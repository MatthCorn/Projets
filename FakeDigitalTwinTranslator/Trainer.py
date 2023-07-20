from FakeDigitalTwinTranslator.Network import TransformerTranslator
from FakeDigitalTwinTranslator.FDTData import LoadData
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from FakeDigitalTwinTranslator.Error import ErrorAction

# Ce script sert à l'apprentissage du réseau Network.TransformerTranslator

# local = r'C:\\Users\\matth\\OneDrive\\Documents\\Python\\Projets'
local = r'C:\Users\Matthieu\Documents\Python\Projets'

d_source = 5
d_target = 5
d_input_Enc = 32
d_input_Dec = 32
d_att = 32
num_flags = 3
num_heads = 4
len_target = 100

# Cette ligne crée les variables globales "TrainingSource", "TrainingTranslation", "ValidationSource" et "ValidationTranslation"
LoadData(ListTypeData=['Training', 'Validation'], len_target=len_target, local=local, variables_dict=vars())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Translator = TransformerTranslator(d_source=d_source, d_target=d_target, d_att=d_att, d_input_Enc=d_input_Enc, target_len=len_target,
                                   d_input_Dec=d_input_Dec, num_flags=num_flags, num_heads=num_heads, device=device)


TrainingEnded = (torch.norm(TrainingTranslation, dim=-1) == 0).unsqueeze(-1).to(torch.float32)

ValidationEnded = (torch.norm(ValidationTranslation, dim=-1) == 0).unsqueeze(-1).to(torch.float32)

batch_size = 50

# Procédure d'entrainement
optimizer = torch.optim.Adam(Translator.parameters(), weight_decay=1e-5, lr=3e-5)
TrainingErrList = []
ValidationErrList = []
for i in tqdm(range(50)):

    TrainingErrList.append(ErrorAction(TrainingSource, TrainingTranslation, TrainingEnded, Translator, batch_size, Action='Training', Optimizer=optimizer))
    ValidationErrList.append(ErrorAction(ValidationSource, ValidationTranslation, ValidationEnded, Translator, batch_size, Action='Validation'))

# torch.save(Translator.state_dict(), os.path.join(local, 'FakeDigitalTwinTranslator', 'Translator'))
# torch.save(optimizer.state_dict(), os.path.join(local, 'FakeDigitalTwinTranslator', 'Optimizer'))


plt.plot(TrainingErrList, 'b')
plt.plot(ValidationErrList, 'r')
plt.show()


