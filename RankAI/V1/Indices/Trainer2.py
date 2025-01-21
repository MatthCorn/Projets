from RankAI.V1.Indices.Ranker import Network, ChoseOutput
from RankAI.V1.Indices.DataMaker import MakeTargetedData
from RankAI.Visualization import MakeGIF
from Complete.LRScheduler import Scheduler
from math import sqrt
from GradObserver.GradObserverClass import DictGradObserver
from Tools.ParamObs import DictParamObserver, RecInitParam
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

################################################################################################################################################
# pour sauvegarder toutes les informations de l'apprentissage
import os
import datetime
from Tools.XMLTools import saveObjAsXml
import pickle

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')
folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
save_path = os.path.join(local, 'RankAI', 'Save', 'V1', 'Indices', folder)
################################################################################################################################################

param = {'n_encoder': 3,
         'path_ini': None,
         # 'path_ini': os.path.join('RankAI', 'Save', 'V1', 'Indices', 'XXXXXXXXXX', 'ParamObs.pkl'),
         'len_in': 10,
         'len_out': 5,
         'd_in': 10,
         'd_att': 64,
         'WidthsEmbedding': [32],
         'n_heads': 4,
         'norm': 'post',
         'dropout': 0,
         'lr': 1e-4,
         'mult_grad': 10000,
         'weight_decay': 1e-3,
         'NDataT': 50000,
         'NDataV': 1000,
         'batch_size': 1000,
         'n_iter': 200,
         'training_strategy': [
             {'mean': [-50, 50], 'std': [0.1, 10]},
             {'mean': [-1000, 1000], 'std': [0.1, 50]},
             {'mean': [-5000, 5000], 'std': [0.1, 50]},
             {'mean': [-10000, 10000], 'std': [0.1, 100]},
         ],
         'distrib': 'uniform',
         'max_lr': 5,
         'FreqGradObs': 1 / 3,
         'warmup': 2}

freq_checkpoint = 1 / 10
nb_frames_GIF = 50
nb_frames_window = int(nb_frames_GIF / len(param['training_strategy']))
res_GIF = 50
n_iter_window = int(param['n_iter'] / len(param['training_strategy']))

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

mini_batch_size = 5000
n_minibatch = int(NDataT / mini_batch_size)
batch_size = param['batch_size']
n_batch = int(mini_batch_size / batch_size)

n_iter = param['n_iter']
TrainingError = []
TrainingPerf = []
ValidationError = []
ValidationPerf = []
PlottingError = []
PlottingPerf = []

n_updates = int(NDataT / batch_size) * n_iter
warmup_steps = int(NDataT / batch_size) * param['warmup']
lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param['max_lr'])

PlottingInput, PlottingOutput = MakeTargetedData(
    NInput=NInput,
    NOutput=NOutput,
    DVec=DVec,
    mean_min=min([window['mean'][0] for window in param['training_strategy']]),
    mean_max=max([window['mean'][1] for window in param['training_strategy']]),
    std_min=min([window['std'][0] for window in param['training_strategy']]),
    std_max=max([window['std'][1] for window in param['training_strategy']]),
    distrib=param['distrib'],
    NData=res_GIF,
    WeightCut=WeightCut,
    WeightSort=WeightSort,
    plot=True,
)

best_state_dict = N.state_dict().copy()

for window in param['training_strategy']:
    TrainingInput, TrainingOutput = MakeTargetedData(
        NInput=NInput,
        NOutput=NOutput,
        DVec=DVec,
        mean_min=window['mean'][0],
        mean_max=window['mean'][1],
        std_min=window['std'][0],
        std_max=window['std'][1],
        distrib=param['distrib'],
        NData=NDataT,
        WeightCut=WeightCut,
        WeightSort=WeightSort,
    )

    ValidationInput, ValidationOutput = MakeTargetedData(
        NInput=NInput,
        NOutput=NOutput,
        DVec=DVec,
        mean_min=window['mean'][0],
        mean_max=window['mean'][1],
        std_min=window['std'][0],
        std_max=window['std'][1],
        distrib=param['distrib'],
        NData=NDataV,
        WeightCut = WeightCut,
        WeightSort = WeightSort,
    )

    base_std = float(torch.std(ValidationOutput.to(torch.float)))

    for j in tqdm(range(n_iter_window)):
        error = 0
        perf = 0
        time_to_observe = (int(j * param['FreqGradObs']) == (j * param['FreqGradObs']))
        time_for_checkpoint = (int(j * freq_checkpoint) == (j * freq_checkpoint))
        time_for_GIF = (j in torch.linspace(0, n_iter_window, nb_frames_window, dtype=int))

        for p in range(n_minibatch):
            InputMiniBatch = TrainingInput[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
            OutputMiniBatch = TrainingOutput[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
            for k in range(n_batch):
                optimizer.zero_grad(set_to_none=True)

                InputBatch = InputMiniBatch[k * batch_size:(k + 1) * batch_size]
                OutputBatch = OutputMiniBatch[k * batch_size:(k + 1) * batch_size]
                Prediction = N(InputBatch)

                err = torch.norm(Prediction - OutputBatch, p=2) / sqrt(batch_size * NOutput) / base_std
                (param['mult_grad'] * err).backward()
                optimizer.step()

                if p == 0 and time_to_observe:
                    DictGrad.update()

                if lr_scheduler is not None:
                    lr_scheduler.step()

                error += float(err) / (n_batch * n_minibatch)
                perf += float(torch.sum(ChoseOutput(Prediction, NOutput) == OutputBatch)) / (NDataT * NOutput)

        if time_to_observe:
            DictGrad.next(j)

        with torch.no_grad():
            Input = ValidationInput.to(device)
            Output = ValidationOutput.to(device)
            Prediction = N(Input)

            err = torch.norm(Prediction - Output, p=2) / sqrt(NDataV * NOutput) / base_std
            ValidationError.append(float(err))
            ValidationPerf.append(float(torch.sum(ChoseOutput(Prediction, NOutput) == Output)) / (NDataV * NOutput))

        TrainingError.append(error)
        TrainingPerf.append(perf)

        if error == min(TrainingError):
            best_state_dict = N.state_dict().copy()

        if time_for_GIF:
            with torch.no_grad():
                Input = PlottingInput.to(device)
                Output = PlottingOutput.to(device)
                Prediction = N(Input)

                err = torch.norm(Prediction - Output, p=2, dim=[-1, -2]) / sqrt(NOutput) / base_std
                perf = torch.sum(ChoseOutput(Prediction, NOutput) == Output, dim=[-1, -2]) / NOutput
                PlottingError.append(err.reshape(res_GIF, res_GIF).tolist())
                PlottingPerf.append(perf.reshape(res_GIF, res_GIF).tolist())

        if time_for_checkpoint:
            try:
                os.mkdir(save_path)
            except:
                pass
            error = {'TrainingError': TrainingError,
                     'ValidationError': ValidationError,
                     'TrainingPerf': TrainingPerf,
                     'ValidationPerf': ValidationPerf,
                     'PlottingError': PlottingError,
                     'PlottingPerf': PlottingPerf}
            saveObjAsXml(param, os.path.join(save_path, 'param'))
            saveObjAsXml(error, os.path.join(save_path, 'error'))
            torch.save(best_state_dict, os.path.join(save_path, 'Best_network'))
            with open(os.path.join(save_path, 'DictGrad.pkl'), 'wb') as file:
                pickle.dump(DictGrad, file)
            with open(os.path.join(save_path, 'ParamObs.pkl'), 'wb') as file:
                ParamObs = DictParamObserver(N)
                pickle.dump(ParamObs, file)

MakeGIF([PlottingError, PlottingPerf], res_GIF, param['training_strategy'], param['n_iter'], param['distrib'], save_path)

if True:
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(TrainingError, 'r', label="Ensemble d'entrainement")
    ax1.plot(ValidationError, 'b', label="Ensemble de Validation")
    ax1.plot([1.] * len(ValidationError), 'black')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right')
    ax1.set_title('V1' + '-' + 'Incides' + " : Erreur")

    ax2.plot(TrainingPerf, 'r', label="Ensemble d'entrainement")
    ax2.plot(ValidationPerf, 'b', label="Ensemble de Validation")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right')
    ax2.set_title('V1' + '-' + 'Incides' + " : Performance")

    fig.tight_layout(pad=1.0)

    plt.show()
