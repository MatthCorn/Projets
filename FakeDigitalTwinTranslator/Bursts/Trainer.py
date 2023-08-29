from FakeDigitalTwinTranslator.Bursts.Network import TransformerTranslator
from FakeDigitalTwinTranslator.Bursts.DataLoader import FDTDataLoader
from Tools.XMLTools import saveObjAsXml
import os
import torch
from tqdm import tqdm
from FakeDigitalTwinTranslator.PlotError import Plot
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
    'batch_size': 1000
}

# Cette ligne crée les variables globales "~TYPE~Source" et "~TYPE~Translation" pour tout ~TYPE~ dans ListTypeData
FDTDataLoader(ListTypeData=['Validation', 'Training', 'Evaluation'], local=local, variables_dict=vars())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Translator = TransformerTranslator(d_source=param['d_source'], d_target=param['d_target'], d_att=param['d_att'], d_input_Enc=param['d_input_Enc'],
                                   target_len=param['len_target'], num_encoders=param['num_encoders'], d_input_Dec=param['d_input_Dec'],
                                   num_flags=param['num_flags'], num_heads=param['num_heads'], num_decoders=param['num_decoders'],
                                   RPR_len_decoder=param['RPR_len_decoder'], NbPDWsMemory=param['NbPDWsMemory'], device=device)


TrainingEnded = (torch.norm(TrainingTranslation[:, param['NbPDWsMemory']:], dim=-1) == 0).unsqueeze(-1).to(torch.float32)

ValidationEnded = (torch.norm(ValidationTranslation[:, param['NbPDWsMemory']:], dim=-1) == 0).unsqueeze(-1).to(torch.float32)

EvaluationEnded = (torch.norm(EvaluationTranslation, dim=-1) == 0).unsqueeze(-1).to(torch.float32)


batch_size = param['batch_size']

# Procédure d'entrainement
optimizer = torch.optim.Adam(Translator.parameters(), weight_decay=1e-5, lr=3e-5)
TrainingErrList = []
TrainingErrTransList = []
TrainingErrActList = []
ValidationErrList = []
ValidationErrTransList = []
ValidationErrActList = []
RealEvaluationList = []
CutEvaluationList = []

for i in tqdm(range(100)):
    Error, ErrAct, ErrTrans = ErrorAction(TrainingSource, TrainingTranslation, TrainingEnded, Translator, batch_size, Action='Training', Optimizer=optimizer)
    TrainingErrList.append(Error)
    TrainingErrActList.append(ErrAct)
    TrainingErrTransList.append(ErrTrans)

    Error, ErrAct, ErrTrans = ErrorAction(ValidationSource, ValidationTranslation, ValidationEnded, Translator, batch_size, Action='Validation')
    ValidationErrList.append(Error)
    ValidationErrActList.append(ErrAct)
    ValidationErrTransList.append(ErrTrans)

    RealError, CutError = ErrorAction(EvaluationSource, EvaluationTranslation, EvaluationEnded, Translator, Action='Evaluation')
    RealEvaluationList.append(RealError)
    CutEvaluationList.append(CutError)

error = {'Training':
             {'ErrList': TrainingErrList, 'ErrTransList': TrainingErrTransList, 'ErrActList': TrainingErrActList},
         'Validation':
             {'ErrList': ValidationErrList, 'ErrTransList': ValidationErrTransList, 'ErrActList': ValidationErrActList}
         'Evaluation':
             {'Real': RealEvaluationList, 'Cut': CutEvaluationList}}

folder = datetime.datetime.now().strftime("%d-%m-%Y__%H-%M")
os.mkdir(os.path.join(local, 'FakeDigitalTwinTranslator', 'Bursts', 'Save', folder))

torch.save(Translator.state_dict(), os.path.join(local, 'FakeDigitalTwinTranslator', 'Bursts', 'Save', folder, 'Translator'))
torch.save(optimizer.state_dict(), os.path.join(local, 'FakeDigitalTwinTranslator', 'Bursts', 'Save', folder, 'Optimizer'))
saveObjAsXml(param, os.path.join(local, 'FakeDigitalTwinTranslator', 'Bursts', 'Save', folder, 'param'))
saveObjAsXml(error, os.path.join(local, 'FakeDigitalTwinTranslator', 'Bursts', 'Save', folder, 'error'))

Plot(os.path.join(local, 'FakeDigitalTwinTranslator', 'Bursts', 'Save', folder, 'error'), eval=True)


