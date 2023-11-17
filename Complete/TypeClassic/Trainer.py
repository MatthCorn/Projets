from Complete.TypeClassic.ClassicNetwork import TransformerTranslator
from Complete.TypeClassic.Error import ErrorAction
from Complete.DataRelated import FDTDataLoader
from Tools.XMLTools import saveObjAsXml
import os
import numpy as np
import torch
from tqdm import tqdm
import datetime
from GitPush import git_push

# Ce script sert à l'apprentissage du réseau Network.TransformerTranslator

local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'OneDrive', 'Documents', 'Python', 'Projets')
# local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'Documents', 'Python', 'Projets')

param = {
    'd_pulse': 5,
    'd_PDW': 5,
    'd_att': 64,
    'n_flags': 3,
    'n_heads': 16,
    'n_encoders': 3,
    'n_decoders': 3,
    'n_trackers': 4,
    'n_PDWs_memory': 10,
    'len_target': 30,
    'len_source': 32,
    'batch_size': 2048
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

translator = TransformerTranslator(d_pulse=param['d_pulse'], d_PDW=param['d_PDW'], d_att=param['d_att'], len_target=param['len_target'],
                                   len_source=param['len_source'], n_encoders=param['n_encoders'], n_flags=param['n_flags'], n_heads=param['n_heads'],
                                   n_decoders=param['n_decoders'], n_PDWs_memory=param['n_PDWs_memory'], device=device, norm='pre')

batch_size = param['batch_size']

# Procédure d'entrainement
TrainingErrList = []
TrainingErrTransList = []
ValidationErrList = []
ValidationErrTransList = []

folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
save_path = os.path.join(local, 'Complete', 'TypeClassic', 'Save', folder)
os.mkdir(save_path)
optimizer = torch.optim.Adam(translator.parameters(), lr=1e-4)

list_dir = os.listdir(os.path.join(local, 'Complete', 'Data'))
list_dir = ['D_0.3']

for dir in list_dir:
    path = os.path.join(local, 'Complete', 'Data', dir)
    validation_source, validation_translation, training_source, training_translation = FDTDataLoader(path=path, len_target=param['len_target'])
    training_ended = training_translation[:, param['n_PDWs_memory']:, param['d_PDW'] + param['n_flags'] + 1].unsqueeze(-1)
    validation_ended = validation_translation[:, param['n_PDWs_memory']:, param['d_PDW'] + param['n_flags'] + 1].unsqueeze(-1)


    # On calcule l'écart type
    std = np.std(training_translation.numpy(), axis=(0, 1))

    n_epochs = 20
    for i in tqdm(range(n_epochs)):
        error, error_trans = ErrorAction(training_source, training_translation, training_ended, translator, batch_size, action='Training', optimizer=optimizer)
        TrainingErrList.append(error)
        # normalisation de l'erreur par rapport à l'écart-type sur chaque coordonnée physique
        error_trans[0], error_trans[1], error_trans[2], error_trans[3], error_trans[4] = \
            error_trans[0]/std[0], error_trans[1]/std[1], error_trans[2]/std[2], error_trans[3]/std[3], error_trans[4]/std[4]
        TrainingErrTransList.append(error_trans)

        translator.eval()
        error, error_trans = ErrorAction(validation_source, validation_translation, validation_ended, translator, batch_size, action='Validation')
        ValidationErrList.append(error)
        # normalisation de l'erreur par rapport à l'écart-type sur chaque coordonnée physique
        error_trans[0], error_trans[1], error_trans[2], error_trans[3], error_trans[4] = \
            error_trans[0]/std[0], error_trans[1]/std[1], error_trans[2]/std[2], error_trans[3]/std[3], error_trans[4]/std[4]
        ValidationErrTransList.append(error_trans)


    error = {'Training':
                 {'ErrList': TrainingErrList, 'ErrTransList': TrainingErrTransList},
             'Validation':
                 {'ErrList': ValidationErrList, 'ErrTransList': ValidationErrTransList}
             }

    os.mkdir(os.path.join(save_path, dir))

    torch.save(translator.state_dict(), os.path.join(save_path, dir, 'Translator'))
    torch.save(optimizer.state_dict(), os.path.join(save_path, dir, 'Optimizer'))
    saveObjAsXml(param, os.path.join(save_path, dir, 'param'))
    saveObjAsXml(error, os.path.join(save_path, dir, 'error'))

    # git_push(local=local, file=os.path.join(save_path, dir), CommitMsg='simu '+folder+dir)



