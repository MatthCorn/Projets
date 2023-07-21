from FakeDigitalTwinTranslator.Network import TransformerTranslator
from FakeDigitalTwinTranslator.DataLoader import FDTDataLoader
from FakeDigitalTwin.XMLTools import saveObjAsXml
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from FakeDigitalTwinTranslator.Error import ErrorAction
import datetime

# Ce script sert à l'apprentissage du réseau Network.TransformerTranslator

local = r'C:\\Users\\matth\\OneDrive\\Documents\\Python\\Projets'
# local = r'C:\Users\Matthieu\Documents\Python\Projets'

param = {
    'd_source' : 5,
    'd_target' : 5,
    'd_input_Enc' : 32,
    'd_input_Dec' : 32,
    'd_att' : 32,
    'num_flags' : 3,
    'num_heads' : 4,
    'num_encoders' : 4,
    'num_decoders' : 4,
    'len_target' : 100,
    'RPR_len_decoder' : 64,
    'batch_size' : 50
}

# Cette ligne crée les variables globales "~TYPE~Source" et "~TYPE~Translation" pour tout ~TYPE~ dans ListTypeData
FDTDataLoader(ListTypeData=['Training', 'Validation', 'Evaluation'], len_target=param['len_target'], local=local, variables_dict=vars())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Translator = TransformerTranslator(d_source=param['d_source'], d_target=param['d_target'], d_att=param['d_att'], d_input_Enc=param['d_input_Enc'],
                                   target_len=param['len_target'], num_encoders=param['num_encoders'], d_input_Dec=param['d_input_Dec'],
                                   num_flags=param['num_flags'], num_heads=param['num_heads'], num_decoders=param['num_decoders'],
                                   RPR_len_decoder=param['RPR_len_decoder'], device=device)


TrainingEnded = (torch.norm(TrainingTranslation, dim=-1) == 0).unsqueeze(-1).to(torch.float32)

ValidationEnded = (torch.norm(ValidationTranslation, dim=-1) == 0).unsqueeze(-1).to(torch.float32)

EvaluationEnded = (torch.norm(EvaluationTranslation, dim=-1) == 0).unsqueeze(-1).to(torch.float32)

batch_size = param['batch_size']

# Procédure d'entrainement
optimizer = torch.optim.Adam(Translator.parameters(), weight_decay=1e-5, lr=3e-5)
TrainingErrList = []
ValidationErrList = []
RealEvalutionList = []
CutEvaluationList = []
for i in tqdm(range(10)):

    TrainingErrList.append(ErrorAction(TrainingSource, TrainingTranslation, TrainingEnded, Translator, batch_size, Action='Training', Optimizer=optimizer))
    ValidationErrList.append(ErrorAction(ValidationSource, ValidationTranslation, ValidationEnded, Translator, batch_size, Action='Validation'))

    RealError, CutError = ErrorAction(EvaluationSource, EvaluationTranslation, EvaluationEnded, Translator, Action='Evaluation')
    RealEvalutionList.append(RealError)
    CutEvaluationList.append(CutError)

error = {'TrainingErrList': TrainingErrList, 'ValidationErrList': ValidationErrList, 'RealEvalutionList': RealEvalutionList, 'CutEvaluationList': CutEvaluationList}

folder = datetime.datetime.now().strftime("%d-%m-%Y__%H-%M")
os.mkdir(os.path.join(local, 'FakeDigitalTwinTranslator', 'Save', folder))

torch.save(Translator.state_dict(), os.path.join(local, 'FakeDigitalTwinTranslator', 'Save', folder, 'Translator'))
torch.save(optimizer.state_dict(), os.path.join(local, 'FakeDigitalTwinTranslator', 'Save', folder, 'Optimizer'))
saveObjAsXml(param, os.path.join(local, 'FakeDigitalTwinTranslator', 'Save', folder, 'param'))
saveObjAsXml(error, os.path.join(local, 'FakeDigitalTwinTranslator', 'Save', folder, 'error'))


fig, ((ax1, ax2)) = plt.subplots(2, 1)
ax1.plot(TrainingErrList, 'b', label="Ensemble d'entrainement"); ax1.plot(ValidationErrList, "r", label="Ensemble de validation");
ax1.set_title("Evolution de l'erreur d'entrainement"); ax1.set_xlabel('Epoch'); ax1.set_ylabel("erreur d'entrainement"); ax1.legend(loc='upper right')
ax2.plot(RealEvalutionList, 'r', label="Erreur sur traduction réelle"); ax2.plot(CutEvaluationList, 'b', label='Erreur sur traduction tronquée');
ax2.set_title("Evolution de l'erreur sur traduction"); ax2.set_xlabel('Epoch'), ax2.set_ylabel('Erreur sur traduction'); ax2.legend(loc='upper right')
plt.show()


