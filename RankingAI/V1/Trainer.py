import numpy as np

from RankingAI.V1.DataMaker import MakeData
from Complete.LRScheduler import Scheduler
from math import sqrt, log10
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.cuda.set_device(2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Relative = False
NVec = 5
DVec = 10

if Relative:
    from RankingAI.V1.Detector import Network
    N = Network(n_encoder=6, d_in=DVec, d_model=64, d_att=64, WidthsEmbedding=[32], num_heads=4, relative=True, masked=False)
else:
    from RankingAI.V1.Ranker import Network
    N = Network(n_encoder=6, max_len=NVec, d_in=DVec, d_att=64, WidthsEmbedding=[32], n_heads=4, norm='post')

N.to(device)

optimizer = torch.optim.Adam(N.parameters(), weight_decay=1e-3, lr=3e-4)

Max = 10
Min = 0

NDataT = 100000
NDataV = 1000

weights = torch.rand(DVec, device=device)
# weights = torch.tensor([1., 0, 0, 0, 0, 0, 0, 0, 0, 0], device=device)
ValidationInput, _, ValidationRanks, ValidationSorted = MakeData(NVec=NVec, DVec=DVec, sigma=1, NData=NDataV, ShiftInterval=[Min, Max], weights=weights)

batch_size = 10000
n_batch = int(NDataT/batch_size)

n_iter = 200

n_updates = int(NDataT / batch_size) * n_iter
warmup_frac = 0.2
warmup_steps = warmup_frac * n_updates
# lr_scheduler = None
lr_scheduler = Scheduler(optimizer, 64, warmup_steps, max=5)

TrainingErrorR = []
TrainingErrorS = []
ValidationErrorR = []
ValidationErrorS = []

fig, (ax1, ax2) = plt.subplots(1, 2)

ShiftList = list(np.logspace(1, log10(Max), 1))
ShiftList = list(map(int, ShiftList))
ShiftIntervals = [[Min, ShiftList[i]] for i in range(len(ShiftList))]

lmd = 1.

for ShiftInterval in ShiftIntervals:
    TrainingInput, _, TrainingRanks, TrainingSorted = MakeData(NVec=NVec, DVec=DVec, sigma=1, NData=NDataT, ShiftInterval=ShiftInterval, weights=weights)

    for j in tqdm(range(n_iter)):
        errorR = 0
        errorS = 0
        for k in range(n_batch):
            optimizer.zero_grad(set_to_none=True)

            InputBatch = TrainingInput[k*batch_size:(k+1)*batch_size].to(device)
            RanksBatch = TrainingRanks[k*batch_size:(k+1)*batch_size].to(device)
            SortedBatch = TrainingSorted[k * batch_size:(k + 1) * batch_size].to(device)
            RanksPrediction, SortedPrediction = N(InputBatch)

            errR = torch.norm(RanksBatch-RanksPrediction, p=2)
            errS = torch.norm(SortedBatch-SortedPrediction, p=2)
            err = lmd * errS + (1 - lmd) * errR
            err.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            errorR += float(errR)/n_batch
            errorS += float(errS)/n_batch

        with torch.no_grad():
            Input = ValidationInput.to(device)
            ValidationRanks = ValidationRanks.to(device)
            ValidationSorted = ValidationSorted.to(device)
            RanksPrediction, SortedPrediction = N(Input)

            errR = torch.norm(ValidationRanks - RanksPrediction, p=2)
            errS = torch.norm(ValidationSorted - SortedPrediction, p=2)
            ValidationErrorR.append(float(errR) / sqrt(NDataV * 5))
            ValidationErrorS.append(float(errS) / sqrt(NDataV * 5))

        TrainingErrorR.append(errorR/sqrt(batch_size*5))
        TrainingErrorS.append(errorS / sqrt(batch_size * 5))


ax1.plot(TrainingErrorR, 'r', label="Ensemble d'entrainement")
ax1.set_title('Erreur ranking')
ax1.plot(ValidationErrorR, 'b', label="Ensemble de Validation")
ax1.legend(loc='upper right')

ax2.plot(TrainingErrorS, 'r', label="Ensemble d'entrainement")
ax2.plot([float(torch.std(ValidationSorted))]*len(TrainingErrorS), 'black')
ax2.set_title('Erreur sorting')
ax2.plot(ValidationErrorS, 'b', label="Ensemble de Validation")
ax2.legend(loc='upper right')

plt.show()

print('end')