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

# Cette ligne crée les variables globales "~TYPE~Source" et "~TYPE~Translation" pour tout ~TYPE~ dans ListTypeData
LoadData(ListTypeData=['Training', 'Validation', 'Evaluation'], len_target=len_target, local=local, variables_dict=vars())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Translator = TransformerTranslator(d_source=d_source, d_target=d_target, d_att=d_att, d_input_Enc=d_input_Enc, target_len=len_target,
                                   d_input_Dec=d_input_Dec, num_flags=num_flags, num_heads=num_heads, device=device)


TrainingEnded = (torch.norm(TrainingTranslation, dim=-1) == 0).unsqueeze(-1).to(torch.float32)

ValidationEnded = (torch.norm(ValidationTranslation, dim=-1) == 0).unsqueeze(-1).to(torch.float32)

EvaluationEnded = (torch.norm(EvaluationTranslation, dim=-1) == 0).unsqueeze(-1).to(torch.float32)

batch_size = 50

# Procédure d'entrainement
optimizer = torch.optim.Adam(Translator.parameters(), weight_decay=1e-5, lr=3e-5)
TrainingErrList = []
ValidationErrList = []
RealEvalutionList = []
CutEvaluationList = []
for i in tqdm(range(1000)):

    TrainingErrList.append(ErrorAction(TrainingSource, TrainingTranslation, TrainingEnded, Translator, batch_size, Action='Training', Optimizer=optimizer))
    ValidationErrList.append(ErrorAction(ValidationSource, ValidationTranslation, ValidationEnded, Translator, batch_size, Action='Validation'))

    RealError, CutError = ErrorAction(EvaluationSource, EvaluationTranslation, EvaluationEnded, Translator, Action='Evaluation')
    RealEvalutionList.append(RealError)
    CutEvaluationList.append(CutError)

torch.save(Translator.state_dict(), os.path.join(local, 'FakeDigitalTwinTranslator', 'Translator-20-07-1000Epochs'))
torch.save(optimizer.state_dict(), os.path.join(local, 'FakeDigitalTwinTranslator', 'Optimizer-20-07-1000Epochs'))

fig, ((ax1, ax2)) = plt.subplots(2, 1)
ax1.plot(TrainingErrList, 'b', label="Ensemble d'entrainement"); ax1.plot(ValidationErrList, "r", label="Ensemble de validation");
ax1.set_title("Evolution de l'erreur d'entrainement"); ax1.set_xlabel('Epoch'); ax1.set_ylabel("erreur d'entrainement"); ax1.legend(loc='upper right')
ax2.plot(RealEvalutionList, 'r', label="Erreur sur traduction réelle"); ax2.plot(CutEvaluationList, 'b', label='Erreur sur traduction tronquée');
ax2.set_title("Evolution de l'erreur sur traduction"); ax2.set_xlabel('Epoch'), ax2.set_ylabel('Erreur sur traduction'); ax2.legend(loc='upper right')
plt.show()


