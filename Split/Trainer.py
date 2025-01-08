from Split.DataMaker import GetData
from Split.Network import TransformerTranslator
from VisualExample import Plot_inplace
from Complete.LRScheduler import Scheduler
from GradObserver.GradObserverClass import DictGradObserver
from Tools.ParamObs import DictParamObserver, RecInitParam
from Tools.XMLTools import loadXmlAsObj
from math import sqrt
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

save = True
################################################################################################################################################
if save:
    # pour sauvegarder toutes les informations de l'apprentissage
    import os
    import datetime
    from Tools.XMLTools import saveObjAsXml
    import pickle

    local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')
    folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
    save_path = os.path.join(local, 'Split', 'Save', folder)
################################################################################################################################################

param = {'n_encoder': 4,
         'n_decoder': 2,
         'len_in_full': 200,
         'len_in_window': 20,
         'len_out': 35,
         'len_out_temp': 8,
         'path_ini': None,
         # 'path_ini': os.path.join('Split', 'Save', '2024-09-23__16-53'),
         'retrain': None,
         # 'retrain': os.path.join('Split', 'Save', '2024-09-23__16-53'),
         'd_att': 64,
         'widths_embedding': [16],
         'n_heads': 4,
         'norm': 'post',
         'dropout': 0,
         'lr': 3e-4,
         'mult_grad': 10000,
         'weight_decay': 1e-3,
         'NDataT': 2000000,
         'NDataV': 5000,
         'batch_size': 1000,
         'n_iter': 10,
         'max_lr': 5,
         'FreqGradObs': 1/3,
         'warmup': 1,
         'dropping_step_list': [-1]}

freq_checkpoint = 1/20

if torch.cuda.is_available():
    torch.cuda.set_device(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

[ValidationInput, ValidationOutput, ValidationMasks], [TrainingInput, TrainingOutput, TrainingMasks] = GetData(
    param['len_in_full'], param['len_in_window'], param['len_out'], param['len_out_temp'],
    param['NDataT'], param['NDataV'], save_path=os.path.join(local, 'Split', 'Data'))

d_in = ValidationInput.size(-1)
d_out = ValidationOutput.size(-1)

N = TransformerTranslator(d_in, d_out, d_att=param['d_att'], n_heads=param['n_heads'], n_encoders=param['n_encoder'],
                          n_decoders=param['n_decoder'], widths_embedding=param['widths_embedding'], len_in=param['len_in_window'],
                          len_out=param['len_out'], norm=param['norm'], dropout=param['dropout'])
DictGrad = DictGradObserver(N)

if param['path_ini'] is not None:
    path_to_load = os.path.join(local, param['path_ini'], 'ParamObs.pkl')
    with open(path_to_load, 'rb') as file:
        ParamObs = pickle.load(file)
    RecInitParam(N, ParamObs)

if param['retrain'] is not None:
    path_to_load = os.path.join(local, param['retrain'], 'Network_weights')
    N.load_state_dict(torch.load(path_to_load, map_location=device))

N.to(device)

optimizer = torch.optim.Adam(N.parameters(), weight_decay=param['weight_decay'], lr=param['lr'])

NDataT = param['NDataT']
NDataV = param['NDataV']
len_out = param['len_out']

mini_batch_size = 10000
n_minibatch = int(NDataT/mini_batch_size)
batch_size = param['batch_size']
n_batch = int(mini_batch_size/batch_size)

n_iter = param['n_iter']
TrainingError = []
ValidationError = []

n_updates = int(NDataT / batch_size) * n_iter
warmup_steps = int(NDataT / batch_size) * param['warmup']
dropping_step_list = list(int(NDataT / batch_size) * torch.tensor(param['dropping_step_list']))
lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param['max_lr'], dropping_step_list=dropping_step_list)

best_stat = N.state_dict().copy()

for j in tqdm(range(n_iter)):
    error = 0
    time_to_observe = (int(j * param['FreqGradObs']) == (j * param['FreqGradObs']))
    time_for_checkpoint = (int(j * freq_checkpoint) == (j * freq_checkpoint))
    for p in range(n_minibatch):
        InputMiniBatch = TrainingInput[p*mini_batch_size:(p+1)*mini_batch_size].to(device)
        OutputMiniBatch = TrainingOutput[p*mini_batch_size:(p+1)*mini_batch_size].to(device)
        MaskMiniBatch = [Mask[p * mini_batch_size:(p + 1) * mini_batch_size].to(device) for Mask in TrainingMasks]

        for k in range(n_batch):
            optimizer.zero_grad(set_to_none=True)

            InputBatch = InputMiniBatch[k*batch_size:(k+1)*batch_size].to(device)
            OutputBatch = OutputMiniBatch[k * batch_size:(k + 1) * batch_size].to(device)
            TargetMaskBatch = [MaskMiniBatch[0][k * batch_size:(k + 1) * batch_size].to(device),
                               MaskMiniBatch[1][k * batch_size:(k + 1) * batch_size].to(device),
                               MaskMiniBatch[2][k * batch_size:(k + 1) * batch_size].to(device)]
            SourceMaskBatch = MaskMiniBatch[3][k * batch_size:(k + 1) * batch_size].to(device)
            Prediction = N(InputBatch, OutputBatch, SourceMaskBatch, TargetMaskBatch)

            err = torch.norm(Prediction[:, :-1, :]-OutputBatch[:, 1:, :], p=2) / sqrt(batch_size*d_out*(len_out-1))
            (param['mult_grad'] * err).backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if k == 0 and time_to_observe:
                DictGrad.update()

            error += float(err) / (n_batch*n_minibatch)

    if time_to_observe:
        DictGrad.next(j)

    with torch.no_grad():
        Input = ValidationInput.to(device)
        Output = ValidationOutput.to(device)
        TargetMasks = [Mask.to(device) for Mask in ValidationMasks[:-1]]
        SourceMask = ValidationMasks[3].to(device)
        Prediction = N(Input, Output, SourceMask, TargetMasks)

        err = torch.norm(Prediction[:, :-1, :] - Output[:, 1:, :], p=2)
        ValidationError.append(float(err) / sqrt(NDataV*d_out*(len_out-1)))

    TrainingError.append(error)

    if error == min(TrainingError):
        best_state_dict = N.state_dict().copy()

    if time_for_checkpoint:
        try:
            os.mkdir(save_path)
        except:
            pass
        error = {'Training': TrainingError,
                 'Validation': ValidationError}
        saveObjAsXml(param, os.path.join(save_path, 'param'))
        saveObjAsXml(error, os.path.join(save_path, 'error'))
        torch.save(best_state_dict, os.path.join(save_path, 'Best_network'))
        with open(os.path.join(save_path, 'DictGrad.pkl'), 'wb') as file:
            pickle.dump(DictGrad, file)
        with open(os.path.join(save_path, 'ParamObs.pkl'), 'wb') as file:
            ParamObs = DictParamObserver(N)
            pickle.dump(ParamObs, file)

################################################################################################################################################
if save:
    try:
        os.mkdir(save_path)
    except:
        pass

    DictGrad.del_module()
    error = {'Training': TrainingError,
             'Validation': ValidationError}
    saveObjAsXml(param, os.path.join(save_path, 'param'))
    saveObjAsXml(error, os.path.join(save_path, 'error'))
    torch.save(N.state_dict(), os.path.join(save_path, 'Network_weights'))
    torch.save(best_state_dict, os.path.join(save_path, 'Best_network'))
    with open(os.path.join(save_path, 'DictGrad.pkl'), 'wb') as file:
        pickle.dump(DictGrad, file)
    with open(os.path.join(save_path, 'ParamObs.pkl'), 'wb') as file:
        ParamObs = DictParamObserver(N)
        pickle.dump(ParamObs, file)
################################################################################################################################################

if True:
    fig, ax1 = plt.subplots()

    ax1.plot(TrainingError, 'r', label="Ensemble d'entrainement")
    ax1.plot(ValidationError, 'b', label="Ensemble de Validation")
    ax1.plot([float(torch.std(ValidationOutput))] * len(ValidationError), 'black')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right')
    ax1.set_title("Erreur au cours de l'apprentissage")

    fig.tight_layout(pad=1.0)

    plt.show()
