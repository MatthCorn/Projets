from StepByStep.S1.Network import TransformerTranslator
from StepByStep.S1.TorchNetwork import Network
from StepByStep.PlotError import Plot
from StepByStep.S1.Error import ErrorAction
from Tools.XMLTools import saveObjAsXml
import os
import torch
import numpy as np
from tqdm import tqdm
import datetime

# Ce script sert à l'apprentissage du réseau Network.TransformerTranslator

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')

param = {
    'd_source': 5,
    'd_target': 5,
    'd_input_Enc': 128,
    'd_input_Dec': 128,
    'd_att': 128,
    'num_flags': 0,
    'num_heads': 8,
    'num_encoders': 3,
    'num_decoders': 3,
    'NbPDWsMemory': 10,
    'len_target': 10,
    'RPR_len_decoder': 10,
    'batch_size': 2048
}

TrainingSource = torch.tensor(np.load(os.path.join(local, 'StepByStep', 'S1', 'Data', 'Training', 'PulsesAnt.npy')))
TrainingTranslation = torch.tensor(np.load(os.path.join(local, 'StepByStep', 'S1', 'Data', 'Training', 'PDWsDCI.npy')))
ValidationSource = torch.tensor(np.load(os.path.join(local, 'StepByStep', 'S1', 'Data', 'Validation', 'PulsesAnt.npy')))
ValidationTranslation = torch.tensor(np.load(os.path.join(local, 'StepByStep', 'S1', 'Data', 'Validation', 'PDWsDCI.npy')))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Translator = TransformerTranslator(d_source=param['d_source'], d_target=param['d_target'], d_att=param['d_att'], d_input_Enc=param['d_input_Enc'],
#                                    target_len=param['len_target'], num_encoders=param['num_encoders'], d_input_Dec=param['d_input_Dec'],
#                                    num_flags=param['num_flags'], num_heads=param['num_heads'], num_decoders=param['num_decoders'],
#                                    RPR_len_decoder=param['RPR_len_decoder'], NbPDWsMemory=param['NbPDWsMemory'], device=device)

Translator = Network(d_model=256, d_source=5, d_target=5, max_len=10, nhead=16, num_decoder_layers=3, num_encoder_layers=3, device=device)


# ValidationTranslation = torch.rand(size=ValidationTranslation.shape)
# TrainingTranslation = torch.rand(size=TrainingTranslation.shape)


batch_size = param['batch_size']

# Procédure d'entrainement
optimizer = torch.optim.Adam(Translator.parameters(), lr=1e-3)
TrainingErrList = []
TrainingErrTransList = []
ValidationErrList = []
ValidationErrTransList = []

ValidationSdt = torch.std(ValidationTranslation, dim=(0, 1))
ValidationSdt[0] = torch.mean(torch.std(ValidationTranslation, dim=1), dim=0)[0]

TrainingSdt = torch.std(TrainingTranslation, dim=(0, 1))
TrainingSdt[0] = torch.mean(torch.std(TrainingTranslation, dim=1), dim=0)[0]


# TrainingSource = TrainingSource/TrainingSdt
# TrainingTranslation = TrainingTranslation/TrainingSdt
# ValidationSource = ValidationSource/ValidationSdt
# ValidationTranslation = ValidationTranslation/ValidationSdt


NbEpochs = 100

ScaleList = [1.]


for i in tqdm(range(NbEpochs)):
    Error, ErrTrans = ErrorAction(TrainingSource, TrainingTranslation, Translator, batch_size, Action='Training', Optimizer=optimizer)
    TrainingErrList.append(Error)
    # TrainingErrTransList.append([el/ScalingFactor for el in ErrTrans])
    TrainingErrTransList.append((torch.tensor(ErrTrans)/TrainingSdt).tolist())

    Error, ErrTrans = ErrorAction(ValidationSource, ValidationTranslation, Translator, batch_size, Action='Validation')
    ValidationErrList.append(Error)
    # ValidationErrTransList.append([el/ScalingFactor for el in ErrTrans])
    ValidationErrTransList.append((torch.tensor(ErrTrans)/ValidationSdt).tolist())


error = {'Training':
             {'ErrList': TrainingErrList, 'ErrTransList': TrainingErrTransList},
         'Validation':
             {'ErrList': ValidationErrList, 'ErrTransList': ValidationErrTransList}}

folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
save_path = os.path.join(local, 'StepByStep', 'S1', 'Save', folder)
os.mkdir(save_path)

saveObjAsXml(error, os.path.join(save_path, 'error'))

Plot(os.path.join(save_path, 'error'), std=False)