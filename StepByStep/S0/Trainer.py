from StepByStep.S0.Network import TransformerTranslator
from StepByStep.S0.TorchNetwork import Network
from StepByStep.S0.TFixup.Network import Network as TNetwork
from StepByStep.PlotError import Plot
from StepByStep.S0.Error import ErrorAction
from Tools.XMLTools import saveObjAsXml
import os
import torch
from tqdm import tqdm
import datetime
import numpy as np

# Ce script sert à l'apprentissage du réseau Network.TransformerTranslator

local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'OneDrive', 'Documents', 'Python', 'Projets')
# local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'Documents', 'Python', 'Projets')

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
    'len_target': 10,
    'RPR_len_decoder': 10,
    'batch_size': 2048
}

# Cette ligne crée les variables globales "~TYPE~Source" et "~TYPE~Translation" pour tout ~TYPE~ dans ListTypeData
TrainingSource = torch.rand(size=(200000, 10, 5))
TrainingTranslation = TrainingSource.clone()

ValidationSource = torch.rand(size=(6000, 10, 5))
ValidationTranslation = ValidationSource.clone()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Translator = Network(d_model=64, d_source=5, d_target=5, max_len=10, nhead=4, num_decoder_layers=3, num_encoder_layers=3, dropout=0.2, device=device)
Translator = TNetwork(d_att=1024, d_source=5, d_target=5, len_target=10, len_source=10, n_heads=4, network_depth=3, device=device, fixup=True)

# ValidationTranslation = torch.rand(size=ValidationTranslation.shape)
# TrainingTranslation = torch.rand(size=TrainingTranslation.shape)


batch_size = param['batch_size']

# Procédure d'entrainement
optimizer = torch.optim.Adam(Translator.parameters(), lr=1e-4)
TrainingErrList = []
TrainingErrTransList = []
ValidationErrList = []
ValidationErrTransList = []

ValidationSdt = torch.std(ValidationTranslation, dim=(0, 1))
ValidationSdt[0] = torch.mean(torch.std(ValidationTranslation, dim=1), dim=0)[0]

TrainingSdt = torch.std(TrainingTranslation, dim=(0, 1))
TrainingSdt[0] = torch.mean(torch.std(TrainingTranslation, dim=1), dim=0)[0]


TrainingSource = TrainingSource/TrainingSdt
TrainingTranslation = TrainingTranslation/TrainingSdt
ValidationSource = ValidationSource/ValidationSdt
ValidationTranslation = ValidationTranslation/ValidationSdt


NbEpochs = 20

ScaleMax = 1
ScaleMin = 1e-2
ScaleList = list(np.logspace(np.log10(ScaleMin), np.log10(ScaleMax), 5))
ScaleList = [1.]

for ScalingFactor in ScaleList:
    for i in tqdm(range(NbEpochs)):
        Error, ErrTrans = ErrorAction(ScalingFactor*TrainingSource, ScalingFactor*TrainingTranslation, Translator, batch_size, Action='Training', Optimizer=optimizer)
        TrainingErrList.append(Error)
        TrainingErrTransList.append([el/ScalingFactor for el in ErrTrans])
        # TrainingErrTransList.append((torch.tensor(ErrTrans)/(ScalingFactor*TrainingSdt)).tolist())

        Error, ErrTrans = ErrorAction(ScalingFactor*ValidationSource, ScalingFactor*ValidationTranslation, Translator, batch_size, Action='Validation')
        ValidationErrList.append(Error)
        ValidationErrTransList.append([el/ScalingFactor for el in ErrTrans])
        # ValidationErrTransList.append((torch.tensor(ErrTrans)/(ScalingFactor*ValidationSdt)).tolist())


error = {'Training':
             {'ErrList': TrainingErrList, 'ErrTransList': TrainingErrTransList},
         'Validation':
             {'ErrList': ValidationErrList, 'ErrTransList': ValidationErrTransList}}

folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
save_path = os.path.join(local, 'StepByStep', 'S0', 'Save', folder)
os.mkdir(save_path)

saveObjAsXml(error, os.path.join(save_path, 'error'))

Plot(os.path.join(save_path, 'error'), std=False)


