from RankingAI.V5.Ranker import Network
from RankingAI.V5.DataMaker import MakeData
from Complete.LRScheduler import Scheduler
from GradObserver.GradObserverClass import DictGradObserver
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
    save_path = os.path.join(local, 'RankingAI', 'Save', 'V5', folder)
    os.mkdir(save_path)
################################################################################################################################################

param = {'n_encoder': 8,
         'len_in': 10,
         'd_in': 10,
         'd_att': 64,
         'WidthsEmbedding': [32],
         'n_heads': 4,
         'norm': 'post',
         'lr': 3e-4,
         'weight_decay': 1e-3,
         'NDataT': 5000000,
         'NDataV': 5000,
         'DVec': 10,
         'NInput': 10,
         'batch_size': 5000,
         'n_iter': 900,
         'max_lr': 2,
         'LenOut': 5,
         'FreqGradObs': 1/20}

if torch.cuda.is_available():
    torch.cuda.set_device(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = Network(n_encoder=param['n_encoder'], len_in=param['len_in'], len_latent=param['LenOut'], d_in=param['d_in'],
            d_att=param['d_att'], WidthsEmbedding=param['WidthsEmbedding'], n_heads=param['n_heads'], norm=param['norm'])
DictGrad = DictGradObserver(N)

N.to(device)

optimizer = torch.optim.Adam(N.parameters(), weight_decay=param['weight_decay'], lr=param['lr'])

NDataT = param['NDataT']
NDataV = param['NDataV']
DVec = param['DVec']
NInput = param['NInput']
NOutput = param['LenOut']
WeightsCut = 2 * torch.rand(DVec) - 1
WeightsSort = 2 * torch.rand(DVec) - 1
ValidationInput, ValidationOutput = MakeData(NInput=NInput, NOutput=NOutput, DVec=DVec, sigma=1, NData=NDataV, WeightsCut=WeightsCut, WeightsSort=WeightsSort)


mini_batch_size = 5000
n_minibatch = int(NDataT/mini_batch_size)
batch_size = param['batch_size']
n_batch = int(mini_batch_size/batch_size)

n_iter = param['n_iter']
TrainingError = []
ValidationError = []

n_updates = int(NDataT / batch_size) * 100
warmup_frac = 0.5
warmup_steps = warmup_frac * n_updates
# warmup_steps = int(NDataT / batch_size) * 100
# lr_scheduler = None
lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param['max_lr'])

TrainingInput, TrainingOutput = MakeData(NInput=NInput, NOutput=NOutput, DVec=DVec, sigma=1, NData=NDataT, WeightsCut=WeightsCut, WeightsSort=WeightsSort)

for j in tqdm(range(n_iter)):
    error = 0
    time_to_observ = (int(j * param['FreqGradObs']) == (j * param['FreqGradObs']))
    for p in range(n_minibatch):
        InputMiniBatch = TrainingInput[p*mini_batch_size:(p+1)*mini_batch_size].to(device)
        OutputMiniBatch = TrainingOutput[p*mini_batch_size:(p+1)*mini_batch_size].to(device)
        for k in range(n_batch):
            optimizer.zero_grad(set_to_none=True)

            InputBatch = InputMiniBatch[k*batch_size:(k+1)*batch_size]
            OutputBatch = OutputMiniBatch[k*batch_size:(k+1)*batch_size]
            Prediction = N(InputBatch)

            err = torch.norm(Prediction-OutputBatch, p=2)
            err.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            if p == 0 and time_to_observ:
                DictGrad.update()

            error += float(err)/(n_batch*n_minibatch)

    if time_to_observ:
        DictGrad.next(j)

    with torch.no_grad():
        Input = ValidationInput.to(device)
        Output = ValidationOutput.to(device)
        Prediction = N(Input)

        err = torch.norm(Prediction - Output, p=2)
        ValidationError.append(float(err)/sqrt(NDataV*DVec*NOutput))

    TrainingError.append(error/sqrt(batch_size*DVec*NOutput))

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

plt.plot(TrainingError, 'r', label="Ensemble d'entrainement")
plt.plot(ValidationError, 'b', label="Ensemble de Validation")
plt.plot([float(torch.std(ValidationOutput))]*len(ValidationError), 'black')
plt.ylim(bottom=0)
plt.legend(loc='upper right')
plt.title('V5')

plt.show()

print('end')