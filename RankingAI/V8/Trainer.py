from RankingAI.V8.Ranker import Network
from RankingAI.V8.DataMaker import MakeData
from Complete.LRScheduler import Scheduler
from GradObserver.GradObserverClass import DictGradObserver
from math import sqrt
import torch
import torch.nn.functional as F
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
    save_path = os.path.join(local, 'RankingAI', 'Save', 'V8', folder)
    os.mkdir(save_path)
################################################################################################################################################

param = {'n_encoder': 1,
         'len_in': 10,
         'd_in': 10,
         'd_att': 64,
         'WidthsEmbedding': [32],
         'n_heads': 4,
         'norm': 'post',
         'lr': 3e-4,
         'weight_decay': 1e-3,
         'NDataT': 50000,
         'NDataV': 1000,
         'DVec': 10,
         'NInput': 10,
         'batch_size': 1000,
         'n_iter': 10,
         'max_lr': 5,
         'LimCut': 0,
         'FreqGradObs': 1/3,
         'type_error': 'MSE'}
# type_error = MSE ou CE

if torch.cuda.is_available():
    torch.cuda.set_device(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = Network(n_encoder=param['n_encoder'], len_in=param['len_in'], d_in=param['d_in'], d_att=param['d_att'],
            WidthsEmbedding=param['WidthsEmbedding'], n_heads=param['n_heads'], norm=param['norm'])
DictGrad = DictGradObserver(N)

N.to(device)

type_error = param['type_error']

optimizer = torch.optim.Adam(N.parameters(), weight_decay=param['weight_decay'], lr=param['lr'])

NDataT = param['NDataT']
NDataV = param['NDataV']
DVec = param['DVec']
LimCut = param['LimCut']
NInput = param['NInput']
WeightsCut = 2 * torch.rand(DVec) - 1
WeightsSort = 2 * torch.rand(DVec) - 1
ValidationInput, ValidationOutput = MakeData(NInput=NInput, LimCut=LimCut, DVec=DVec, sigma=1, NData=NDataV, WeightsCut=WeightsCut, WeightsSort=WeightsSort)

mini_batch_size = 50000
n_minibatch = int(NDataT/mini_batch_size)
batch_size = param['batch_size']
n_batch = int(mini_batch_size/batch_size)

n_iter = param['n_iter']
TrainingError = []
TrainingClassError = []
ValidationError = []
ValidationClassError = []

n_updates = int(NDataT / batch_size) * n_iter
# warmup_frac = 0.2
# warmup_steps = warmup_frac * n_updates
warmup_steps = 100
lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param['max_lr'])

TrainingInput, TrainingOutput = MakeData(NInput=10, LimCut=0, DVec=10, sigma=1, NData=NDataT, WeightsCut=WeightsCut, WeightsSort=WeightsSort)

for j in tqdm(range(n_iter)):
    error = 0
    error_class = 0
    time_to_observ = (int(j * param['FreqGradObs']) == (j * param['FreqGradObs']))
    for p in range(n_minibatch):
        InputMiniBatch = TrainingInput[p*mini_batch_size:(p+1)*mini_batch_size].to(device)
        OutputMiniBatch = TrainingOutput[p*mini_batch_size:(p+1)*mini_batch_size].to(device)
        for k in range(n_batch):
            optimizer.zero_grad(set_to_none=True)

            InputBatch = InputMiniBatch[k*batch_size:(k+1)*batch_size]
            OutputBatch = OutputMiniBatch[k*batch_size:(k+1)*batch_size]
            Prediction = N(InputBatch)

            if type_error == 'CE':
                err = torch.norm(Prediction - OutputBatch)/sqrt(batch_size*DVec*NInput)
            if type_error == 'MSE':
                err = torch.nn.functional.mse_loss(Prediction, OutputBatch)
            err.backward()
            optimizer.step()

            if p == 0 and time_to_observ:
                DictGrad.update()

            ClassPrediction = F.one_hot(Prediction.argmax(dim=-1), num_classes=Prediction.size(-1))
            ErrorClassTraining = torch.norm(OutputBatch - ClassPrediction, p=2) / sqrt(2 * batch_size * NInput)

            if lr_scheduler is not None:
                lr_scheduler.step()

            error += float(err)/(n_batch*n_minibatch)
            error_class += float(ErrorClassTraining)/(n_batch*n_minibatch)

    if time_to_observ:
        DictGrad.next(j)

    with torch.no_grad():
        Input = ValidationInput.to(device)
        Output = ValidationOutput.to(device)
        Prediction = N(Input)

        if type_error == 'CE':
            err = torch.norm(Prediction - Output)/sqrt(NDataV*DVec*NInput)
        if type_error == 'MSE':
            err = torch.nn.functional.mse_loss(Prediction, Output)

        ValidationError.append(float(err))
        ClassPrediction = F.one_hot(Prediction.argmax(dim=-1), num_classes=Prediction.size(-1))
        ErrorClassValidation = torch.norm(Output - ClassPrediction, p=2)/sqrt(2 * NDataV * NInput)
        ValidationClassError.append(float(ErrorClassValidation))

    TrainingError.append(error)
    TrainingClassError.append(float(error_class))

################################################################################################################################################
if save:
    DictGrad.del_module()
    error = {'Training': TrainingError,
             'Validation': ValidationError}
    # Sauvegarde des informations de l'entrainement
    # torch.save(N.state_dict(), os.path.join(save_path, 'Translator'))
    # torch.save(optimizer.state_dict(), os.path.join(save_path, 'Optimizer'))
    saveObjAsXml(param, os.path.join(save_path, 'param'))
    saveObjAsXml(error, os.path.join(save_path, 'error'))
    with open(os.path.join(save_path, 'DictGrad.pkl'), 'wb') as file:
        pickle.dump(DictGrad, file)
################################################################################################################################################

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(TrainingError, 'r', label="Ensemble d'entrainement")
ax1.set_title('Erreur minimisée')
ax1.set_ylim(bottom=0)
ax1.plot(ValidationError, 'b', label="Ensemble de Validation")
ax1.legend(loc='upper right')

ax2.plot(TrainingClassError, 'r', label="Ensemble d'entrainement")
ax2.set_title('Erreur de classification')
ax2.set_ylim(bottom=0)
ax2.plot(ValidationClassError, 'b', label="Ensemble de Validation")
ax2.legend(loc='upper right')

fig.suptitle('V8')

plt.show()

print('end')