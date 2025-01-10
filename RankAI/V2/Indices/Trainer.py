from RankAI.V2.Indices.Ranker import Network, ChoseOutput
from RankAI.V2.Indices.DataMaker import MakeData
from Complete.LRScheduler import Scheduler
from math import sqrt
from GradObserver.GradObserverClass import DictGradObserver
from Tools.ParamObs import DictParamObserver, RecInitParam
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
    save_path = os.path.join(local, 'RankAI', 'Save', 'V2', 'Indices', folder)
################################################################################################################################################

param = {'n_encoder': 3,
         'path_ini': None,
         # 'path_ini': os.path.join('RankAI', 'Save', 'V2', 'Indices', 'XXXXXXXXXX', 'ParamObs.pkl'),
         'len_in': 10,
         'len_out': 5,
         'd_in': 10,
         'd_att': 64,
         'WidthsEmbedding': [32],
         'n_heads': 4,
         'norm': 'post',
         'dropout': 0,
         'lr': 3e-4,
         'mult_grad': 10000,
         'weight_decay': 1e-3,
         'NDataT': 50000,
         'NDataV': 1000,
         'batch_size': 1000,
         'n_iter': 50,
         'max_lr': 5,
         'FreqGradObs': 1/3,
         'warmup': 2}

if torch.cuda.is_available():
    torch.cuda.set_device(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = Network(n_encoder=param['n_encoder'], len_in=param['len_in'], len_latent=param['len_out'], d_in=param['d_in'], d_att=param['d_att'],
            WidthsEmbedding=param['WidthsEmbedding'], n_heads=param['n_heads'], norm=param['norm'], dropout=param['dropout'])
DictGrad = DictGradObserver(N)

if param['path_ini'] is not None:
    path_to_load = os.path.join(local, param['path_ini'])
    with open(path_to_load, 'rb') as file:
        ParamObs = pickle.load(file)
    RecInitParam(N, ParamObs)

N.to(device)

optimizer = torch.optim.Adam(N.parameters(), weight_decay=param['weight_decay'], lr=param['lr'])

NDataT = param['NDataT']
NDataV = param['NDataV']
DVec = param['d_in']
NInput = param['len_in']
NOutput = param['len_out']
WeightCut = 2 * torch.rand(DVec) - 1
WeightCut = WeightCut / torch.norm(WeightCut)
WeightSort = 2 * torch.rand(DVec) - 1
WeightSort = WeightSort / torch.norm(WeightSort)
ValidationInput, ValidationOutput = MakeData(NInput=NInput, NOutput=NOutput, DVec=DVec, sigma=1, NData=NDataV, WeightCut=WeightCut, WeightSort=WeightSort)

mini_batch_size = 5000
n_minibatch = int(NDataT/mini_batch_size)
batch_size = param['batch_size']
n_batch = int(mini_batch_size/batch_size)

n_iter = param['n_iter']
TrainingError = []
TrainingPerf = []
ValidationError = []
ValidationPerf = []

n_updates = int(NDataT / batch_size) * n_iter
warmup_steps = int(NDataT / batch_size) * param['warmup']
lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param['max_lr'])

TrainingInput, TrainingOutput = MakeData(NInput=NInput, NOutput=NOutput, DVec=DVec, sigma=1, NData=NDataT, WeightCut=WeightCut, WeightSort=WeightSort)

for j in tqdm(range(n_iter)):
    error = 0
    perf = 0
    time_to_observe = (int(j * param['FreqGradObs']) == (j * param['FreqGradObs']))
    for p in range(n_minibatch):
        InputMiniBatch = TrainingInput[p*mini_batch_size:(p+1)*mini_batch_size].to(device)
        OutputMiniBatch = TrainingOutput[p*mini_batch_size:(p+1)*mini_batch_size].to(device)
        for k in range(n_batch):
            optimizer.zero_grad(set_to_none=True)

            InputBatch = InputMiniBatch[k*batch_size:(k+1)*batch_size]
            OutputBatch = OutputMiniBatch[k*batch_size:(k+1)*batch_size]
            Prediction = N(InputBatch)

            err = torch.norm(Prediction-OutputBatch, p=2) / sqrt(batch_size*NOutput)
            (param['mult_grad'] * err).backward()
            optimizer.step()

            if p == 0 and time_to_observe:
                DictGrad.update()

            if lr_scheduler is not None:
                lr_scheduler.step()

            error += float(err)/(n_batch*n_minibatch)
            perf += float(torch.sum(ChoseOutput(Prediction, NOutput) == OutputBatch))/(NDataT*NOutput)

    if time_to_observe:
        DictGrad.next(j)

    with torch.no_grad():
        Input = ValidationInput.to(device)
        Output = ValidationOutput.to(device)
        Prediction = N(Input)

        err = torch.norm(Prediction - Output, p=2)/sqrt(NDataV*NOutput)
        ValidationError.append(float(err))
        ValidationPerf.append(float(torch.sum(ChoseOutput(Prediction, NOutput) == Output))/(NDataV*NOutput))

    TrainingError.append(error)
    TrainingPerf.append(perf)

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
             'ValidationError': ValidationError,
             'TrainingPerf': TrainingPerf,
             'ValidationPerf': ValidationPerf}
    saveObjAsXml(param, os.path.join(save_path, 'param'))
    saveObjAsXml(error, os.path.join(save_path, 'error'))
    with open(os.path.join(save_path, 'DictGrad.pkl'), 'wb') as file:
        pickle.dump(DictGrad, file)
    with open(os.path.join(save_path, 'ParamObs.pkl'), 'wb') as file:
        ParamObs = DictParamObserver(N)
        pickle.dump(ParamObs, file)
################################################################################################################################################

if True:
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(TrainingError, 'r', label="Ensemble d'entrainement")
    ax1.plot(ValidationError, 'b', label="Ensemble de Validation")
    ax1.plot([float(torch.std(ValidationOutput.to(torch.float)))] * len(ValidationError), 'black')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right')
    ax1.set_title('V2' + '-' + 'Incides' + " : Erreur")

    ax2.plot(TrainingPerf, 'r', label="Ensemble d'entrainement")
    ax2.plot(ValidationPerf, 'b', label="Ensemble de Validation")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right')
    ax2.set_title('V2' + '-' + 'Incides' + " : Performance")

    fig.tight_layout(pad=1.0)

    plt.show()
