from RankAI.V1.Vecteurs.DataMaker import MakeData
from RankAI.V1.Vecteurs.Ranker import Network, ChoseOutput
from Complete.LRScheduler import Scheduler
from GradObserver.GradObserverClass import DictGradObserver
from Tools.ParamObs import DictParamObserver, RecInitParam
from math import sqrt
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


param = {'n_encoder': 5,
         'len_in': 10,
         'len_out': 5,
         'path_ini': None,
         # 'path_ini': os.path.join('RankAI', 'Save', 'V1', 'Vecteurs', 'XXXXXXXXXX', 'ParamObs.pkl'),
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
         'n_iter': 50,
         'max_lr': 5,
         'FreqGradObs': 1/30,
         'warmup': 2}

if torch.cuda.is_available():
    torch.cuda.set_device(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = Network(n_encoder=param['n_encoder'], len_in=param['len_in'], len_latent=param['len_out'], d_in=param['d_in'], d_att=param['d_att'],
            WidthsEmbedding=param['WidthsEmbedding'], n_heads=param['n_heads'], norm=param['norm'], dropout=param['dropout'])
DictGrad = DictGradObserver(N)


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

mini_batch_size = 50000
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
            OutputBatch = OutputMiniBatch[k * batch_size:(k + 1) * batch_size]
            Prediction = N(InputBatch)

            err = torch.norm(Prediction-OutputBatch, p=2) / sqrt(batch_size*DVec*NOutput)
            (param['mult_grad'] * err).backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if k == 0 and time_to_observe:
                DictGrad.update()

            error += float(err)/(n_batch*n_minibatch)
            perf += float(torch.sum(ChoseOutput(Prediction, InputBatch) == ChoseOutput(OutputBatch, InputBatch)))/(NDataT*NOutput)

    if time_to_observe:
        DictGrad.next(j)

    with torch.no_grad():
        Input = ValidationInput.to(device)
        Output = ValidationOutput.to(device)
        Prediction = N(Input)

        err = torch.norm(Prediction - Output, p=2) / sqrt(NDataV*DVec*NOutput)
        ValidationError.append(float(err))
        ValidationPerf.append(float(torch.sum(ChoseOutput(Prediction, Input) == ChoseOutput(Output, Input)))/(NDataV*NOutput))

    TrainingError.append(error)
    TrainingPerf.append(perf)



