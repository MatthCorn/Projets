from Complete.TypeTrackerInspired.TrackerNetwork import TransformerTranslator
from Complete.TypeTrackerInspired.Hookers import Hookers
from Complete.TypeTrackerInspired.Error import ErrorAction
from Complete.DataRelated import FDTDataLoader
from Tools.XMLTools import saveObjAsXml
import os
import torch
from tqdm import tqdm
from Complete.PlotError import Plot
import datetime
from GitPush import git_push

# Ce script sert à l'apprentissage du réseau Network.TransformerTranslator

local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'OneDrive', 'Documents', 'Python', 'Projets')
# local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'Documents', 'Python', 'Projets')

param = {
    'd_pulse': 5,
    'd_PDW': 5,
    'd_att': 32,
    'n_flags': 3,
    'n_heads': 4,
    'n_encoders': 3,
    'n_decoders': 3,
    'n_trackers': 4,
    'n_PDWs_memory': 10,
    'len_target': 20,
    'batch_size': 2048
}

weights = {
    'trans': 1,
    'act': 1,
    'var': {'mod': 1, 'threshold': 5},
    'div': {'mod': 1, 'threshold': 0.1875},
    'sym': 1,
    'likeli': 1,
}

validation_source, validation_translation, training_source, training_translation = FDTDataLoader(local=local)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

translator = TransformerTranslator(d_pulse=param['d_pulse'], d_PDW=param['d_PDW'], d_att=param['d_att'], len_target=param['len_target'],
                                    n_encoders=param['n_encoders'], n_tracker=param['n_trackers'], n_flags=param['n_flags'],
                                    n_heads=param['n_heads'], n_decoders=param['n_decoders'], n_PDWs_memory=param['n_PDWs_memory'], device=device)


hookers = Hookers(translator)


training_ended = (torch.norm(training_translation[:, param['n_PDWs_memory']:], dim=-1) == 0).unsqueeze(-1).to(torch.float32)

validation_ended = (torch.norm(validation_translation[:, param['n_PDWs_memory']:], dim=-1) == 0).unsqueeze(-1).to(torch.float32)

batch_size = param['batch_size']

# Procédure d'entrainement
optimizer = torch.optim.Adam(translator.parameters(), lr=3e-4)
TrainingErrList = []
TrainingErrTransList = []
TrainingErrActList = []
ValidationErrList = []
ValidationErrTransList = []
ValidationErrActList = []

n_epochs = 100
for i in tqdm(range(n_epochs)):
    error, error_act, error_trans = ErrorAction(training_source, training_translation, training_ended, translator, hookers, weights, batch_size, action='Training', optimizer=optimizer)
    TrainingErrList.append(error)
    TrainingErrActList.append(error_act)
    TrainingErrTransList.append(error_trans)

    translator.eval()
    error, error_act, error_trans = ErrorAction(validation_source, validation_translation, validation_ended, translator, hookers, weights, batch_size, action='Validation')
    ValidationErrList.append(error)
    ValidationErrActList.append(error_act)
    ValidationErrTransList.append(error_trans)


error = {'Training':
             {'ErrList': TrainingErrList, 'ErrTransList': TrainingErrTransList, 'ErrActList': TrainingErrActList},
         'Validation':
             {'ErrList': ValidationErrList, 'ErrTransList': ValidationErrTransList, 'ErrActList': ValidationErrActList}}

folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
save_path = os.path.join(local, 'Complete', 'Save', folder)
os.mkdir(save_path)

torch.save(translator.state_dict(), os.path.join(save_path, 'Translator'))
torch.save(optimizer.state_dict(), os.path.join(save_path, 'Optimizer'))
saveObjAsXml(param, os.path.join(save_path, 'param'))
saveObjAsXml(error, os.path.join(save_path, 'error'))

# git_push(local=local, file=save_path, CommitMsg='simu '+folder)

Plot(os.path.join(save_path, 'error'), std=True)


