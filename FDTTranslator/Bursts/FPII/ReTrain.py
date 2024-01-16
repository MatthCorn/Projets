from FDTTranslator.Bursts.Network import TransformerTranslator
from FDTTranslator.Bursts.DataLoader import FDTDataLoader
from Tools.XMLTools import saveObjAsXml, loadXmlAsObj
import os
import torch
from tqdm import tqdm
from FDTTranslator.PlotError import Plot
from FDTTranslator.Bursts.Error import ErrorAction
import datetime
from GitPush import git_push

# Ce script sert à l'apprentissage du réseau Network.TransformerTranslator

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projet')

folder = '2023-09-07__10-30'
save_path = os.path.join(local, 'FDTTranslator', 'Bursts', 'FPII', 'Save', folder)

param = loadXmlAsObj(os.path.join(save_path, 'param'))

# Cette ligne crée les variables globales "~TYPE~Source" et "~TYPE~Translation" pour tout ~TYPE~ dans ListTypeData
FDTDataLoader(ListTypeData=['Validation', 'Training', 'Evaluation'], local=local, variables_dict=vars(), TypeBursts='FPII')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Translator = TransformerTranslator(d_source=param['d_source'], d_target=param['d_target'], d_att=param['d_att'], d_input_Enc=param['d_input_Enc'],
                                   target_len=param['len_target'], num_encoders=param['num_encoders'], d_input_Dec=param['d_input_Dec'],
                                   num_flags=param['num_flags'], num_heads=param['num_heads'], num_decoders=param['num_decoders'],
                                   RPR_len_decoder=param['RPR_len_decoder'], NbPDWsMemory=param['NbPDWsMemory'], device=device)

Translator.load_state_dict(torch.load(os.path.join(save_path, 'Translator')))
Translator.train()

TrainingEnded = (torch.norm(TrainingTranslation[:, param['NbPDWsMemory']:], dim=-1) == 0).unsqueeze(-1).to(torch.float32)

ValidationEnded = (torch.norm(ValidationTranslation[:, param['NbPDWsMemory']:], dim=-1) == 0).unsqueeze(-1).to(torch.float32)

EvaluationEnded = (torch.norm(EvaluationTranslation, dim=-1) == 0).unsqueeze(-1).to(torch.float32)


batch_size = param['batch_size']

# Procédure d'entrainement
optimizer = torch.optim.Adam(Translator.parameters(), lr=3e-4)
optimizer.load_state_dict(torch.load(os.path.join(save_path, 'Optimizer')))

error = loadXmlAsObj(os.path.join(save_path, 'error'))

TrainingErrList = error['Training']['ErrList']
TrainingErrTransList = error['Training']['ErrTransList']
TrainingErrActList = error['Training']['ErrActList']
ValidationErrList = error['Validation']['ErrList']
ValidationErrTransList = error['Validation']['ErrTransList']
ValidationErrActList = error['Validation']['ErrActList']
RealEvaluationList = error['Evaluation']['Real']
CutEvaluationList = error['Evaluation']['Cut']

for i in tqdm(range(50)):
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
             {'ErrList': ValidationErrList, 'ErrTransList': ValidationErrTransList, 'ErrActList': ValidationErrActList},
         'Evaluation':
             {'Real': RealEvaluationList, 'Cut': CutEvaluationList}}

folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
os.mkdir(save_path)

torch.save(Translator.state_dict(), os.path.join(save_path, 'Translator'))
torch.save(optimizer.state_dict(), os.path.join(save_path, 'Optimizer'))
saveObjAsXml(param, os.path.join(save_path, 'param'))
saveObjAsXml(error, os.path.join(save_path, 'error'))

git_push(local=local, file=save_path, CommitMsg='simu '+folder)

Plot(os.path.join(save_path, 'error'), eval=True)


