from Complete.TypeTrackerInspired.TrackerNetwork import TransformerTranslator
from Complete.TypeTrackerInspired.Hookers import Hookers
from Complete.TypeTrackerInspired.Error import ErrorAction
from Complete.DataRelated import FDTDataLoader, GetStd
from Tools.XMLTools import saveObjAsXml
import numpy as np
import os
import torch
from tqdm import tqdm
import datetime
from Complete.LRScheduler import Scheduler
from Tools.GitPush import git_push

# Ce script sert à l'apprentissage du réseau Network.TransformerTranslator

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')

param = {
    'd_pulse': 5,
    'd_pulse_buffed': 14,
    'd_PDW': 5,
    'd_PDW_buffed': 14,
    'd_att': 64,
    'n_flags': 3,
    'n_heads': 4,
    'n_encoders': 3,
    'n_decoders': 3,
    'n_PDWs_memory': 10,
    'len_target': 30,
    'len_source': 32,
    'batch_size': 256,
    'threshold': 7,
    'freq_ech': 3,
    'n_trackers': 4,
    'is_RoPE': True
}

weights_hookers = {
    'trans': 1,
    'var': {'mod': 0, 'threshold': 5},
    'div': {'mod': 0, 'threshold': 0.1875},
    'sym': 0.,
    'likeli': 0
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def TrainWithHookers(weights_hookers, n_epochs=50):

    translator = TransformerTranslator(d_pulse=param['d_pulse'], d_pulse_buffed=param['d_pulse_buffed'], d_PDW=param['d_PDW'], d_PDW_buffed=param['d_PDW_buffed'], d_att=param['d_att'],
                                       len_target=param['len_target'], freq_ech=param['freq_ech'], weights=os.path.join(local, 'Complete', 'Weights'), norm='pre',
                                       len_source=param['len_source'], n_encoders=param['n_encoders'], n_decoders=param['n_decoders'], n_flags=param['n_flags'], is_RoPE=param['is_RoPE'],
                                       n_heads=param['n_heads'], n_PDWs_memory=param['n_PDWs_memory'], device=device, threshold=param['threshold'], n_trackers=param['n_trackers'])

    print('nombre de paramètres : ', sum(p.numel() for p in translator.parameters() if p.requires_grad))

    hookers = Hookers(translator)

    batch_size = param['batch_size']

    # Procédure d'entrainement
    TrainingErrList = []
    TrainingErrTransList = []
    ValidationErrList = []
    ValidationErrTransList = []

    folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
    save_path = os.path.join(local, 'Complete', 'TypeTrackerInspired', 'Save', folder)
    os.mkdir(save_path)

    size = 'High_Interaction'

    # list_dir = os.listdir(os.path.join(local, 'Complete', 'Data', size))
    list_dir = ['D_5']

    weights_error = torch.tensor((np.load(os.path.join(local, 'Complete', 'Weights', 'output_std.npy')) + 1e-10)**-1, device=device)

    alt_rep = torch.eye(11)
    alt_rep[-5:-3, -5:-3] = torch.tensor([[1 / 2, 1 / 2], [1 / 2, -1 / 2]])
    alt_rep = alt_rep.to(device)

    for dir in list_dir:
        optimizer = torch.optim.Adam(translator.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

        path = os.path.join(local, 'Complete', 'Data', size, dir)
        validation_source, validation_translation, training_source, training_translation = FDTDataLoader(path=path, len_target=param['len_target'], device=device)
        training_ended = training_translation[:, param['n_PDWs_memory']:, param['d_PDW'] + param['n_flags'] + 1].unsqueeze(-1)
        validation_ended = validation_translation[:, param['n_PDWs_memory']:, param['d_PDW'] + param['n_flags'] + 1].unsqueeze(-1)

        n_updates = int(len(training_source)/param['batch_size']) * n_epochs
        warmup_frac = 0.2
        warmup_steps = warmup_frac * n_updates
        lr_scheduler = None
        # lr_scheduler = Scheduler(optimizer, param['d_att'], warmup_steps, max=50)

        # On calcule l'écart type
        # std = np.std(torch.matmul(training_translation, alt_rep.t().to(torch.device('cpu'))).numpy(), axis=(0, 1))
        std = GetStd(torch.matmul(training_translation[:, param['n_PDWs_memory']:].cpu(), alt_rep.t().cpu()))

        for i in tqdm(range(n_epochs)):
            error, error_trans = ErrorAction(training_source, training_translation, training_ended, translator, weights_error=weights_error,
                                             batch_size=batch_size, action='Training', optimizer=optimizer, alt_rep=alt_rep[:8, :8],
                                             lr_scheduler=lr_scheduler, weights_hookers=weights_hookers, hookers=hookers)
            TrainingErrList.append(error)
            # normalisation de l'erreur par rapport à l'écart-type sur chaque coordonnée physique
            error_trans[0], error_trans[1], error_trans[2], error_trans[3], error_trans[4] = \
                error_trans[0]/std[0], error_trans[1]/std[1], error_trans[2]/std[2], error_trans[3]/std[3], error_trans[4]/std[4]
            TrainingErrTransList.append(error_trans)

            translator.eval()
            error, error_trans = ErrorAction(validation_source, validation_translation, validation_ended, translator,
                                             batch_size=batch_size, action='Validation', alt_rep=alt_rep[:8, :8],
                                             weights_hookers=weights_hookers, hookers=hookers)
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

    saveObjAsXml(weights_hookers, os.path.join(save_path, 'weights_hookers'))

if __name__ == '__main__':
    list_weights_hookers = [{
    'trans': 1,
    'var': {'mod': 0, 'threshold': 5},
    'div': {'mod': 0, 'threshold': 0.1875},
    'sym': i/200.,
    'likeli': 0
} for i in range(1)]
    
    for weights_hookers in list_weights_hookers:
        TrainWithHookers(weights_hookers=weights_hookers, n_epochs=15)
