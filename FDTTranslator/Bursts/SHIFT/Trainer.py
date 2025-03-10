from FDTTranslator.Bursts.Network import TransformerTranslator
from FDTTranslator.Bursts.TrackerInspired.Network import TransformerTranslator as TTT
from FDTTranslator.Bursts.TorchNetwork import Network
from FDTTranslator.Bursts.DataLoader import FDTDataLoader
from Tools.XMLTools import saveObjAsXml
import os
import torch
from tqdm import tqdm
from FDTTranslator.PlotError import Plot, PlotPropError
from FDTTranslator.Bursts.Error import ErrorAction, DetailObjectiveError
import datetime
import numpy as np
from math import log10
from Tools.GitPush import git_push

# Ce script sert à l'apprentissage du réseau Network.TransformerTranslator

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')

param = {
    'd_source': 5,
    'd_target': 5,
    'd_input_Enc': 32,
    'd_input_Dec': 32,
    'd_att': 32,
    'num_flags': 3,
    'num_heads': 4,
    'num_encoders': 3,
    'num_decoders': 3,
    'NbPDWsMemory': 10,
    'len_target': 20,
    'RPR_len_decoder': 16,
    'batch_size': 2048
}

# Cette ligne crée les variables globales "~TYPE~Source" et "~TYPE~Translation" pour tout ~TYPE~ dans ListTypeData
FDTDataLoader(ListTypeData=['Validation', 'Training'], local=local, variables_dict=vars(), TypeBursts='SHIFT')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Translator = TransformerTranslator(d_source=param['d_source'], d_target=param['d_target'], d_att=param['d_att'], d_input_Enc=param['d_input_Enc'],
#                                    target_len=param['len_target'], num_encoders=param['num_encoders'], d_input_Dec=param['d_input_Dec'],
#                                    num_flags=param['num_flags'], num_heads=param['num_heads'], num_decoders=param['num_decoders'],
#                                    RPR_len_decoder=param['RPR_len_decoder'], NbPDWsMemory=param['NbPDWsMemory'], device=device)

Translator = Network(d_source=param['d_source'], d_target=param['d_target']+param['num_flags'], d_model=param['d_att'], max_len=param['len_target'], nhead=param['num_heads'],
                     num_encoder_layers=param['num_encoders'], num_decoder_layers=param['num_decoders'], dim_feedforward=2048, dropout=0, NbPDWsMemory=param['NbPDWsMemory'],
                     device=device)

# Translator = TTT(d_pulse=param['d_source'], d_PDW=param['d_target'], d_latent=param['d_att'], num_heads=param['num_heads'], num_encoders=param['num_encoders'],
#                  n_trackers=4, num_decoders=param['num_decoders'], NbPDWsMemory=param['NbPDWsMemory'], target_len=param['len_target'], num_flags=param['num_flags'],
#                  device=device, FPIC=False)


TrainingEnded = (torch.norm(TrainingTranslation[:, param['NbPDWsMemory']:], dim=-1) == 0).unsqueeze(-1).to(torch.float32)

ValidationEnded = (torch.norm(ValidationTranslation[:, param['NbPDWsMemory']:], dim=-1) == 0).unsqueeze(-1).to(torch.float32)

# EvaluationEnded = (torch.norm(EvaluationTranslation, dim=-1) == 0).unsqueeze(-1).to(torch.float32)


batch_size = param['batch_size']

# Procédure d'entrainement
optimizer = torch.optim.Adam(Translator.parameters(), lr=3e-4)
TrainingErrList = []
TrainingErrTransList = []
TrainingErrActList = []
ValidationErrList = []
ValidationErrTransList = []
ValidationErrActList = []
# RealEvaluationList = []
# CutEvaluationList = []

NbEpochs = 50
# NbEvalProp = 10
# ListEvalPropErrorId = list(map(int, list(np.logspace(0, log10(NbEpochs), NbEvalProp))))
# DictEvalPropError = {}

for i in tqdm(range(NbEpochs)):
    Error, ErrAct, ErrTrans = ErrorAction(TrainingSource, TrainingTranslation, TrainingEnded, Translator, batch_size, Action='Training', Optimizer=optimizer)
    TrainingErrList.append(Error)
    TrainingErrActList.append(ErrAct)
    TrainingErrTransList.append(ErrTrans)

    Error, ErrAct, ErrTrans = ErrorAction(ValidationSource, ValidationTranslation, ValidationEnded, Translator, batch_size, Action='Validation')
    ValidationErrList.append(Error)
    ValidationErrActList.append(ErrAct)
    ValidationErrTransList.append(ErrTrans)

    # RealError, CutError = ErrorAction(EvaluationSource, EvaluationTranslation, EvaluationEnded, Translator, Action='Evaluation')
    # RealEvaluationList.append(RealError)
    # CutEvaluationList.append(CutError)

    # if i in ListEvalPropErrorId:
    #     DictEvalPropError[str(i)] = DetailObjectiveError(EvaluationSource, EvaluationTranslation, EvaluationEnded, Translator)

error = {'Training':
             {'ErrList': TrainingErrList, 'ErrTransList': TrainingErrTransList, 'ErrActList': TrainingErrActList},
         'Validation':
             {'ErrList': ValidationErrList, 'ErrTransList': ValidationErrTransList, 'ErrActList': ValidationErrActList}}
         # , 'Evaluation':
         #     {'Real': RealEvaluationList, 'Cut': CutEvaluationList}}

folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
save_path = os.path.join(local, 'FDTTranslator', 'Bursts', 'SHIFT', 'Save', folder)
os.mkdir(save_path)

torch.save(Translator.state_dict(), os.path.join(save_path, 'Translator'))
torch.save(optimizer.state_dict(), os.path.join(save_path, 'Optimizer'))
saveObjAsXml(param, os.path.join(save_path, 'param'))
saveObjAsXml(error, os.path.join(save_path, 'error'))
# saveObjAsXml(DictEvalPropError, os.path.join(save_path, 'PropError'))

# git_push(local=local, file=save_path, CommitMsg='simu '+folder)

Plot(os.path.join(save_path, 'error'), eval=False, std=True)
# PlotPropError(os.path.join(save_path, 'PropError'))


