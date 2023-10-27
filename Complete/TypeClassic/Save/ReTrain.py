from Complete.TypeClassic.ClassicNetwork import TransformerTranslator
from Complete.TypeClassic.Error import ErrorAction
from Complete.DataRelated import FDTDataLoader
from Tools.XMLTools import saveObjAsXml, loadXmlAsObj
import os
import numpy as np
import torch
from tqdm import tqdm
import datetime
from GitPush import git_push

# Ce script sert à l'apprentissage du réseau Network.TransformerTranslator

# local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'OneDrive', 'Documents', 'Python', 'Projets')
local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'Documents', 'Python', 'Projets')

folder = os.path.join('2023-10-27__16-12', 'D_3')
save_path = os.path.join(local, 'Complete', 'TypeClassic', 'Save', folder)

param = loadXmlAsObj(os.path.join(save_path, 'param'))

path = os.path.join(local, 'Complete', 'Data', os.path.split(folder)[-1])
# Cette ligne crée les variables globales "~TYPE~Source" et "~TYPE~Translation" pour tout ~TYPE~ dans ListTypeData
validation_source, validation_translation, training_source, training_translation = FDTDataLoader(path=path, len_target=param['len_target'])
training_ended = training_translation[:, param['n_PDWs_memory']:, param['d_PDW'] + param['n_flags'] + 1].unsqueeze(-1)
validation_ended = validation_translation[:, param['n_PDWs_memory']:, param['d_PDW'] + param['n_flags'] + 1].unsqueeze(-1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

translator = TransformerTranslator(d_pulse=param['d_pulse'], d_PDW=param['d_PDW'], d_att=param['d_att'], len_target=param['len_target'],
                                   len_source=param['len_source'], n_encoders=param['n_encoders'], n_flags=param['n_flags'], n_heads=param['n_heads'],
                                   n_decoders=param['n_decoders'], n_PDWs_memory=param['n_PDWs_memory'], device=device)

translator.load_state_dict(torch.load(os.path.join(save_path, 'Translator')))
translator.train()

batch_size = param['batch_size']

# Procédure d'entrainement
optimizer = torch.optim.Adam(translator.parameters(), lr=3e-3)
optimizer.load_state_dict(torch.load(os.path.join(save_path, 'Optimizer')))

error = loadXmlAsObj(os.path.join(save_path, 'error'))

TrainingErrList = error['Training']['ErrList']
TrainingErrTransList = error['Training']['ErrTransList']
ValidationErrList = error['Validation']['ErrList']
ValidationErrTransList = error['Validation']['ErrTransList']

# On calcule l'écart type
std = np.std(training_translation.numpy(), axis=(0, 1))

n_epoch = 150
for i in tqdm(range(n_epoch)):
    error, error_trans = ErrorAction(training_source, training_translation, training_ended, translator, batch_size,
                                     action='Training', optimizer=optimizer)
    TrainingErrList.append(error)
    # normalisation de l'erreur par rapport à l'écart-type sur chaque coordonnée physique
    error_trans[0], error_trans[1], error_trans[2], error_trans[3], error_trans[4] = \
        error_trans[0] / std[0], error_trans[1] / std[1], error_trans[2] / std[2], error_trans[3] / std[3], error_trans[
            4] / std[4]
    TrainingErrTransList.append(error_trans)

    translator.eval()
    error, error_trans = ErrorAction(validation_source, validation_translation, validation_ended, translator,
                                     batch_size, action='Validation')
    ValidationErrList.append(error)
    # normalisation de l'erreur par rapport à l'écart-type sur chaque coordonnée physique
    error_trans[0], error_trans[1], error_trans[2], error_trans[3], error_trans[4] = \
        error_trans[0] / std[0], error_trans[1] / std[1], error_trans[2] / std[2], error_trans[3] / std[3], error_trans[
            4] / std[4]
    ValidationErrTransList.append(error_trans)

error = {'Training':
             {'ErrList': TrainingErrList, 'ErrTransList': TrainingErrTransList},
         'Validation':
             {'ErrList': ValidationErrList, 'ErrTransList': ValidationErrTransList}}

folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
save_path = os.path.join(save_path, folder)
os.mkdir(save_path)

torch.save(translator.state_dict(), os.path.join(save_path, 'Translator'))
torch.save(optimizer.state_dict(), os.path.join(save_path, 'Optimizer'))
saveObjAsXml(param, os.path.join(save_path, 'param'))
saveObjAsXml(error, os.path.join(save_path, 'error'))

# git_push(local=local, file=save_path, CommitMsg='simu '+folder)



