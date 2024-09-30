from RankingAI.V4.DataMaker import MakeData
from math import sqrt
import torch
from tqdm import tqdm
from GradObserver.GradObserverClass import DictGradObserver
import matplotlib.pyplot as plt
from RankingAI.V4.Ranker import Network
from Complete.LRScheduler import Scheduler

save = False
################################################################################################################################################
if save:
    # pour sauvegarder toutes les informations de l'apprentissage
    import os
    import datetime
    from Tools.XMLTools import saveObjAsXml
    import pickle

    local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')
    folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
    save_path = os.path.join(local, 'RankingAI', 'Save', 'V4', folder)
    os.mkdir(save_path)
################################################################################################################################################

param = {'n_encoder': 1,
         'len_in': 10,
         'len_latent': 5,
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
         'LenOut': 5,
         'FreqGradObs': 1/3}

if torch.cuda.is_available():
    torch.cuda.set_device(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NVecIn = param['NInput']
NVecOut = param['LenOut']
DVec = param['DVec']

N = Network(n_encoder=param['n_encoder'], len_in=NVecIn, len_latent=NVecOut, d_in=DVec, d_att=param['d_att'],
            WidthsEmbedding=param['WidthsEmbedding'], n_heads=param['n_heads'], norm=param['norm'])
DictGrad = DictGradObserver(N)

N.to(device)

optimizer = torch.optim.Adam(N.parameters(), weight_decay=param['weight_decay'], lr=param['lr'])

NDataT = param['NDataT']
NDataV = param['NDataV']

weights = torch.rand(DVec, device=device)
# weights = torch.tensor([1., 0, 0, 0, 0, 0, 0, 0, 0, 0], device=device)
ValidationInput, ValidationOutput = MakeData(NInput=NVecIn, NOutput=NVecOut, DVec=DVec, sigma=1, NData=NDataV, weights=weights)

batch_size = param['batch_size']
n_batch = int(NDataT/batch_size)

n_iter = param['n_iter']
TrainingError = []
ValidationError = []

n_updates = int(NDataT / batch_size) * n_iter
warmup_frac = 0.2
warmup_steps = warmup_frac * n_updates
# lr_scheduler = None
lr_scheduler = Scheduler(optimizer, 64, warmup_steps, max=param['max_lr'])

TrainingInput, TrainingOutput = MakeData(NInput=NVecIn, NOutput=NVecOut, DVec=DVec, sigma=1, NData=NDataT, weights=weights)

for j in tqdm(range(n_iter)):
    error = 0
    time_to_observ = (int(j * param['FreqGradObs']) == (j * param['FreqGradObs']))
    for k in range(n_batch):
        optimizer.zero_grad(set_to_none=True)

        InputBatch = TrainingInput[k*batch_size:(k+1)*batch_size].to(device)
        OutputBatch = TrainingOutput[k * batch_size:(k + 1) * batch_size].to(device)
        Prediction = N(InputBatch)


        err = torch.norm(Prediction-OutputBatch, p=2)
        err.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if k == 0 and time_to_observ:
            DictGrad.update()

        error += float(err)/sqrt(batch_size*DVec*NVecOut)

    if time_to_observ:
        DictGrad.next(j)

    with torch.no_grad():
        Input = ValidationInput.to(device)
        Output = ValidationOutput.to(device)
        Prediction = N(Input)

        err = torch.norm(Prediction - Output, p=2)
        ValidationError.append(float(err) / sqrt(NDataV*DVec*NVecOut))

    TrainingError.append(error/n_batch)

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
plt.plot([float(torch.std(ValidationOutput))]*len(TrainingError), 'black')
plt.ylim(bottom=0)
plt.legend(loc='upper right')
plt.title('V4')

plt.show()

print('end')