from RankAI.V0.OneHot.DataMaker import MakeTargetedData
from RankAI.GifCreator import MakeGIF
from RankAI.V0.OneHot.Ranker import Network, ChoseOutput
from Complete.LRScheduler import Scheduler
from GradObserver.GradObserverClass import DictGradObserver
from Tools.ParamObs import DictParamObserver, RecInitParam
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
    save_path = os.path.join(local, 'RankAI', 'Save', 'V0', 'OneHot', folder)
################################################################################################################################################

param = {'n_encoder': 5,
         'len_in': 10,
         'path_ini': None,
         # 'path_ini': os.path.join('RankAI', 'Save', 'V0', 'Vecteurs', 'XXXXXXXXXX', 'ParamObs.pkl'),
         'd_in': 10,
         'd_att': 128,
         'WidthsEmbedding': [32],
         'n_heads': 4,
         'norm': 'post',
         'dropout': 0,
         'lr': 3e-4,
         'mult_grad': 10000,
         'weight_decay': 1e-3,
         'NDataT': 500000,
         'NDataV': 1000,
         'batch_size': 1000,
         'n_iter': 800,
         'training_strategy': [
             {'mean': [-50, 50], 'std': [0.1, 10]},
             {'mean': [-200, 200], 'std': [0.1, 50]},
             {'mean': [-500, 500], 'std': [0.1, 50]},
             {'mean': [-800, 800], 'std': [0.1, 50]},
             {'mean': [-1000, 1000], 'std': [0.1, 50]},
             {'mean': [-1000, 1000], 'std': [0.1, 80]},
             {'mean': [-1000, 1000], 'std': [0.1, 100]},
             {'mean': [-1500, 1500], 'std': [0.1, 100]},
             {'mean': [-2000, 2000], 'std': [0.1, 100]},
             {'mean': [-2500, 2500], 'std': [0.1, 100]},
             {'mean': [-3000, 3000], 'std': [0.1, 100]},
             {'mean': [-4000, 4000], 'std': [0.1, 100]},
             {'mean': [-5000, 5000], 'std': [0.1, 100]},
             {'mean': [-7000, 7000], 'std': [0.1, 100]},
             {'mean': [-9000, 9000], 'std': [0.1, 100]},
             {'mean': [-10000, 10000], 'std': [0.1, 100]},
         ],
         'distrib': 'uniform',
         'max_lr': 5,
         'FreqGradObs': 1/3,
         'warmup': 2,
         'type_error': 'CE'}

freq_checkpoint = 1/10
nb_frames_GIF = 20
nb_frames_window = int(nb_frames_GIF / len(param['training_strategy']))
n_iter_window = int(param['n_iter'] / len(param['training_strategy']))

if torch.cuda.is_available():
    torch.cuda.set_device(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = Network(n_encoder=param['n_encoder'], len_in=param['len_in'], d_in=param['d_in'], d_att=param['d_att'],
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
NVec = param['len_in']
Weight = 2 * torch.rand(DVec) - 1
Weight = Weight / torch.norm(Weight)

mini_batch_size = 50000
n_minibatch = int(NDataT/mini_batch_size)
batch_size = param['batch_size']
n_batch = int(mini_batch_size/batch_size)

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
    NVec=NVec,
    DVec=DVec,
    mean_min=min([window['mean'][0] for window in param['training_strategy']]),
    mean_max=max([window['mean'][1] for window in param['training_strategy']]),
    sigma_min=min([window['std'][0] for window in param['training_strategy']]),
    sigma_max=max([window['std'][1] for window in param['training_strategy']]),
    distrib=param['distrib'],
    NData=100,
    Weight=Weight,
    plot=True,
)

best_state_dict = N.state_dict().copy()

for window in param['training_strategy']:
    TrainingInput, TrainingOutput = MakeTargetedData(
        NVec=NVec,
        DVec=DVec,
        mean_min=window['mean'][0],
        mean_max=window['mean'][1],
        sigma_min=window['std'][0],
        sigma_max=window['std'][1],
        distrib=param['distrib'],
        NData=NDataT,
        Weight=Weight,
    )

    ValidationInput, ValidationOutput = MakeTargetedData(
        NVec=NVec,
        DVec=DVec,
        mean_min=window['mean'][0],
        mean_max=window['mean'][1],
        sigma_min=window['std'][0],
        sigma_max=window['std'][1],
        distrib=param['distrib'],
        NData=NDataV,
        Weight=Weight,
    )

    if param['type_error'] == 'CE':
        base_value = float(torch.log(torch.tensor(NVec)))
    elif param['type_error'] == 'MSE':
        base_value = float(torch.std(ValidationOutput.to(torch.float)))


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

                if param['type_error'] == 'CE':
                    err = torch.nn.functional.cross_entropy(Prediction, OutputBatch) / base_value
                elif param['type_error'] == 'MSE':
                    err = torch.norm(Prediction - OutputBatch, p=2) / sqrt(batch_size * NVec * NVec) / base_value
                (param['mult_grad'] * err).backward()
                optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()

                if k == 0 and time_to_observe:
                    DictGrad.update()

                error += float(err) / (n_batch * n_minibatch)
                perf += float(torch.sum(ChoseOutput(Prediction) == ChoseOutput(OutputBatch))) / (NDataT * NVec)

        if time_to_observe:
            DictGrad.next(j)

        with torch.no_grad():
            Input = ValidationInput.to(device)
            Output = ValidationOutput.to(device)
            Prediction = N(Input)

            if param['type_error'] == 'CE':
                err = torch.nn.functional.cross_entropy(Prediction, Output) / base_value
            elif param['type_error'] == 'MSE':
                err = torch.norm(Prediction - Output, p=2) / sqrt(NDataV*NVec*NVec) / base_value

            ValidationError.append(float(err))
            ValidationPerf.append(float(torch.sum(ChoseOutput(Prediction) == ChoseOutput(Output)))/(NDataV*NVec))

        TrainingError.append(error)
        TrainingPerf.append(perf)

        if error == min(TrainingError):
            best_state_dict = N.state_dict().copy()

        if time_for_checkpoint:
            try:
                os.mkdir(save_path)
            except:
                pass
            error = {'TrainingError': TrainingError,
                     'ValidationError': ValidationError,
                     'TrainingPerf': TrainingPerf,
                     'ValidationPerf': ValidationPerf}
            saveObjAsXml(param, os.path.join(save_path, 'param'))
            saveObjAsXml(error, os.path.join(save_path, 'error'))
            torch.save(best_state_dict, os.path.join(save_path, 'Best_network'))
            with open(os.path.join(save_path, 'DictGrad.pkl'), 'wb') as file:
                pickle.dump(DictGrad, file)
            with open(os.path.join(save_path, 'ParamObs.pkl'), 'wb') as file:
                ParamObs = DictParamObserver(N)
                pickle.dump(ParamObs, file)

        if time_for_GIF:
            with torch.no_grad():
                Input = PlottingInput.to(device)
                Output = PlottingOutput.to(device)
                Prediction = N(Input)

                if param['type_error'] == 'CE':
                    err = torch.mean(torch.nn.functional.cross_entropy(Prediction, Output, reduction='none'), dim=-1) / base_value
                elif param['type_error'] == 'MSE':
                    err = torch.norm(Prediction - Output, p=2, dim=[-1, -2]) / sqrt(NVec * NVec) / base_value

                err = torch.norm(Prediction - Output, p=2, dim=[-1, -2]) / sqrt(NVec) / base_value
                perf = torch.sum(ChoseOutput(Prediction,) == ChoseOutput(Output), dim=[-1]) / NVec
                PlottingError.append(err)
                PlottingPerf.append(perf)

MakeGIF([PlottingError, PlottingPerf], 100, param['training_strategy'], param['distrib'], save_path)

if True:
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(TrainingError, 'r', label="Ensemble d'entrainement")
    ax1.plot(ValidationError, 'b', label="Ensemble de Validation")
    ax1.plot([1.] * len(ValidationError), 'black')

    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right')
    ax1.set_title('V0' + '-' + 'OneHot' + " : Erreur")

    ax2.plot(TrainingPerf, 'r', label="Ensemble d'entrainement")
    ax2.plot(ValidationPerf, 'b', label="Ensemble de Validation")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right')
    ax2.set_title('V0' + '-' + 'OneHot' + " : Performance")

    fig.tight_layout(pad=1.0)

    plt.show()
