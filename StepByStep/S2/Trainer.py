from StepByStep.S1.Network import TransformerTranslator
from StepByStep.S1.DataLoader import FDTDataLoader
from StepByStep.PlotError import Plot
from StepByStep.S1.Error import ErrorAction
from Tools.XMLTools import saveObjAsXml
import os
import torch
from tqdm import tqdm
import datetime

# Ce script sert à l'apprentissage du réseau Network.TransformerTranslator

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')

param = {
    'd_source': 5,
    'd_target': 5,
    'd_input_Enc': 32,
    'd_input_Dec': 32,
    'd_att': 32,
    'num_flags': 0,
    'num_heads': 4,
    'num_encoders': 3,
    'num_decoders': 3,
    'NbPDWsMemory': 10,
    'len_target': 20,
    'RPR_len_decoder': 16,
    'batch_size': 2048
}

# Cette ligne crée les variables globales "~TYPE~Source" et "~TYPE~Translation" pour tout ~TYPE~ dans ListTypeData
FDTDataLoader(ListTypeData=['Validation', 'Training'], local=local, variables_dict=vars())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Translator = TransformerTranslator(d_source=param['d_source'], d_target=param['d_target'], d_att=param['d_att'], d_input_Enc=param['d_input_Enc'],
                                   target_len=param['len_target'], num_encoders=param['num_encoders'], d_input_Dec=param['d_input_Dec'],
                                   num_flags=param['num_flags'], num_heads=param['num_heads'], num_decoders=param['num_decoders'],
                                   RPR_len_decoder=param['RPR_len_decoder'], NbPDWsMemory=param['NbPDWsMemory'], device=device)


batch_size = param['batch_size']

# Procédure d'entrainement
optimizer = torch.optim.Adam(Translator.parameters(), lr=3e-4)
TrainingErrList = []
TrainingErrTransList = []
ValidationErrList = []
ValidationErrTransList = []

NbEpochs = 100

for i in tqdm(range(NbEpochs)):
    Error, ErrTrans = ErrorAction(TrainingSource, TrainingTranslation, Translator, batch_size, Action='Training', Optimizer=optimizer)
    TrainingErrList.append(Error)
    TrainingErrTransList.append(ErrTrans)

    Error, ErrTrans = ErrorAction(ValidationSource, ValidationTranslation, Translator, batch_size, Action='Validation')
    ValidationErrList.append(Error)
    ValidationErrTransList.append(ErrTrans)


error = {'Training':
             {'ErrList': TrainingErrList, 'ErrTransList': TrainingErrTransList},
         'Validation':
             {'ErrList': ValidationErrList, 'ErrTransList': ValidationErrTransList}}

folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
save_path = os.path.join(local, 'StepByStep', 'S1', 'Save', folder)
os.mkdir(save_path)

saveObjAsXml(error, os.path.join(save_path, 'error'))

Plot(os.path.join(save_path, 'error'), std=True)


