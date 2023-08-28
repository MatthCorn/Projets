from FakeDigitalTwinTranslator.Bursts.Network import TransformerTranslator
from FakeDigitalTwinTranslator.Bursts.DataLoader import FDTDataLoader
from FakeDigitalTwin.XMLTools import saveObjAsXml
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from FakeDigitalTwinTranslator.Bursts.Error import ErrorAction
import datetime

# Ce script sert à l'apprentissage du réseau Network.TransformerTranslator

local = r'C:\\Users\\matth\\OneDrive\\Documents\\Python\\Projets'
# local = r'C:\Users\Matthieu\Documents\Python\Projets'

param = {
    'd_source': 5,
    'd_target': 5,
    'd_input_Enc': 16,
    'd_input_Dec': 16,
    'd_att': 16,
    'num_flags': 3,
    'num_heads': 4,
    'num_encoders': 2,
    'num_decoders': 2,
    'NbPDWsMemory': 10,
    'len_target': 20,
    'RPR_len_decoder': 16,
    'batch_size': 10000
}

# Cette ligne crée les variables globales "~TYPE~Source" et "~TYPE~Translation" pour tout ~TYPE~ dans ListTypeData
FDTDataLoader(ListTypeData=['Validation', 'Training'], local=local, variables_dict=vars())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Translator = TransformerTranslator(d_source=param['d_source'], d_target=param['d_target'], d_att=param['d_att'], d_input_Enc=param['d_input_Enc'],
                                   target_len=param['len_target'], num_encoders=param['num_encoders'], d_input_Dec=param['d_input_Dec'],
                                   num_flags=param['num_flags'], num_heads=param['num_heads'], num_decoders=param['num_decoders'],
                                   RPR_len_decoder=param['RPR_len_decoder'], NbPDWsMemory=param['NbPDWsMemory'], device=device)


TrainingEnded = (torch.norm(TrainingTranslation[:, param['NbPDWsMemory']:], dim=-1) == 0).unsqueeze(-1).to(torch.float32)

ValidationEnded = (torch.norm(ValidationTranslation[:, param['NbPDWsMemory']:], dim=-1) == 0).unsqueeze(-1).to(torch.float32)

batch_size = param['batch_size']

# Procédure d'entrainement
optimizer = torch.optim.Adam(Translator.parameters(), weight_decay=1e-5, lr=3e-5)
TrainingErrList = []
TrainingErrTransList = []
TrainingErrActList = []
ValidationErrList = []
ValidationErrTransList = []
ValidationErrActList = []

for i in tqdm(range(5)):
    Error, ErrAct, ErrTrans = ErrorAction(TrainingSource, TrainingTranslation, TrainingEnded, Translator, batch_size, Action='Training', Optimizer=optimizer)
    TrainingErrList.append(Error)
    TrainingErrActList.append(ErrAct)
    TrainingErrTransList.append(ErrTrans)

    Error, ErrAct, ErrTrans = ErrorAction(ValidationSource, ValidationTranslation, ValidationEnded, Translator, batch_size, Action='Validation')
    ValidationErrList.append(Error)
    ValidationErrActList.append(ErrAct)
    ValidationErrTransList.append(ErrTrans)


error = {'Training':
             {'ErrList': TrainingErrList, 'ErrTransList': TrainingErrTransList, 'ErrActList': TrainingErrActList},
         'Validation':
             {'ErrList': ValidationErrList, 'ErrTransList': ValidationErrTransList, 'ErrActList': ValidationErrActList}}

folder = datetime.datetime.now().strftime("%d-%m-%Y__%H-%M")
os.mkdir(os.path.join(local, 'FakeDigitalTwinTranslator', 'Bursts', 'Save', folder))

torch.save(Translator.state_dict(), os.path.join(local, 'FakeDigitalTwinTranslator', 'Bursts', 'Save', folder, 'Translator'))
torch.save(optimizer.state_dict(), os.path.join(local, 'FakeDigitalTwinTranslator', 'Bursts', 'Save', folder, 'Optimizer'))
saveObjAsXml(param, os.path.join(local, 'FakeDigitalTwinTranslator', 'Bursts', 'Save', folder, 'param'))
saveObjAsXml(error, os.path.join(local, 'FakeDigitalTwinTranslator', 'Bursts', 'Save', folder, 'error'))


fig, ((ax11, ax12), (ax21, ax22), (ax31, ax32), (ax41, ax42)) = plt.subplots(4, 2)

ax11.plot(TrainingErrList, 'r', label="Ensemble d'entrainement"); ax11.set_title('Erreur gobale')
ax11.plot(ValidationErrList, 'b', label="Ensemble de Validation"); ax11.legend(loc='upper right')

ax12.plot([el[0] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement"); ax12.set_title('Erreur sur TOA')
ax12.plot([el[0] for el in ValidationErrTransList], 'b', label="Ensemble de Validation"); ax12.legend(loc='upper right')

ax21.plot([el[1] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement"); ax21.set_title('Erreur sur LI')
ax21.plot([el[1] for el in ValidationErrTransList], 'b', label="Ensemble de Validation"); ax21.legend(loc='upper right')

ax22.plot([el[2] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement"); ax22.set_title('Erreur sur Niveau')
ax22.plot([el[2] for el in ValidationErrTransList], 'b', label="Ensemble de Validation"); ax22.legend(loc='upper right')

ax31.plot([el[3] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement"); ax31.set_title('Erreur sur FreqMin')
ax31.plot([el[3] for el in ValidationErrTransList], 'b', label="Ensemble de Validation"); ax31.legend(loc='upper right')

ax32.plot([el[4] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement"); ax32.set_title('Erreur sur FreqMax')
ax32.plot([el[4] for el in ValidationErrTransList], 'b', label="Ensemble de Validation"); ax32.legend(loc='upper right')

ax41.plot(TrainingErrActList, 'r', label="Ensemble d'entrainement"); ax41.set_title("Erreur sur l'action")
ax41.plot(ValidationErrActList, 'b', label="Ensemble de Validation"); ax41.legend(loc='upper right')

plt.show()


