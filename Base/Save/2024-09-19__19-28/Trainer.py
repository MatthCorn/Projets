from Base.DataMaker import GetData
from Base.Network import TransformerTranslator
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
    save_path = os.path.join(local, 'Base', 'Save', folder)
################################################################################################################################################

param = {'n_encoder': 10,
         'n_decoder': 8,
         'len_in': 20,
         'len_out': 25,
         'path_ini': None,
         # 'path_ini': os.path.join('Base', 'Save', '2024-09-23__16-53'),
         # 'retrain': None,
         'retrain': os.path.join('Base', 'Save', '2024-09-23__16-53'),
         'd_att': 128,
         'widths_embedding': [32],
         'n_heads': 4,
         'norm': 'post',
         'dropout': 0,
         'lr': 1e-5,
         'weight_decay': 1e-3,
         'NDataT': 20000,
         'NDataV': 5000,
         'batch_size': 1000,
         'n_iter': 10,
         'max_lr': 5,
         'FreqGradObs': 1/3,
         'warmup': 1}

if torch.cuda.is_available():
    torch.cuda.set_device(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

[ValidationInput, ValidationOutput, ValidationMasks], [TrainingInput, TrainingOutput, TrainingMasks] = GetData(
    param['len_in'], param['len_out'], param['NDataT'], param['NDataV'], save_path=os.path.join(local, 'Base', 'Data'))

d_in = ValidationInput.size(-1)
d_out = ValidationOutput.size(-1)

N = TransformerTranslator(d_in, d_out, d_att=param['d_att'], n_heads=param['n_heads'], n_encoders=param['n_encoder'],
                          n_decoders=param['n_decoder'], widths_embedding=param['widths_embedding'], len_in=param['len_in'],
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
lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param['max_lr'])


for j in tqdm(range(n_iter)):
    error = 0
    time_to_observe = (int(j * param['FreqGradObs']) == (j * param['FreqGradObs']))
    for p in range(n_minibatch):
        InputMiniBatch = TrainingInput[p*mini_batch_size:(p+1)*mini_batch_size].to(device)
        OutputMiniBatch = TrainingOutput[p*mini_batch_size:(p+1)*mini_batch_size].to(device)

        for k in range(n_batch):
            optimizer.zero_grad(set_to_none=True)

            InputBatch = InputMiniBatch[k*batch_size:(k+1)*batch_size].to(device)
            OutputBatch = OutputMiniBatch[k * batch_size:(k + 1) * batch_size].to(device)
            TargetMaskBatch = [TargetMask[0][k * batch_size:(k + 1) * batch_size].to(device),
                               TargetMask[1][k * batch_size:(k + 1) * batch_size].to(device)]
            Prediction = N(InputBatch, OutputBatch, TargetMaskBatch)[:, :-1, :]

            err = torch.norm(Prediction-OutputBatch, p=2)/sqrt(batch_size*d_out*len_out)
            err.backward()
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
        TargetMask = [ValidationMasks[0].to(device), ValidationMasks[1].to(device)]
        Prediction = N(Input, Output, TargetMask)[:, :-1, :]

        err = torch.norm(Prediction - Output, p=2)
        ValidationError.append(float(err) / sqrt(NDataV*d_out*len_out))

    TrainingError.append(error)

################################################################################################################################################
if save:
    try:
        os.mkdir(save_path)
    except:
        for file in os.listdir(save_path):
            os.remove(os.path.join(save_path, file))
        os.rmdir(save_path)
        os.mkdir(save_path)

    DictGrad.del_module()
    error = {'TrainingError': TrainingError,
             'ValidationError': ValidationError}
    saveObjAsXml(param, os.path.join(save_path, 'param'))
    saveObjAsXml(error, os.path.join(save_path, 'error'))
    torch.save(N.state_dict(), os.path.join(save_path, 'Network_weights'))
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
