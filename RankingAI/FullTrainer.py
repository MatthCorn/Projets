import numpy as np

from RankingAI.Ranker import Network
# from RankingAI.Detector import Network

from RankingAI.DataMaker import MakeData
from math import sqrt, log10
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.cuda.set_device(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = Network(n_encoder=6, max_len=5, d_in=10, d_att=64, WidthsEmbedding=[32], n_heads=4)
# N = Network(n_encoder=6, d_in=10, d_model=64, d_att=64, WidthsEmbedding=[32], num_heads=4, relative=True, masked=False)

N.to(device)

optimizer = torch.optim.Adam(N.parameters(), weight_decay=1e-3, lr=3e-4)

Max = 10
Min = 0

NDataT = 200000
NDataV = 1000
DVec = 10
weights = torch.rand(DVec, device=device)
# weights = torch.tensor([1., 0, 0, 0, 0, 0, 0, 0, 0, 0], device=device)
ValidationInput, _, ValidationRanks, ValidationSorted = MakeData(NVec=5, DVec=DVec, sigma=1, NData=NDataV, ShiftInterval=[Min, Max], weights=weights, device=device)

batch_size = 10000
n_batch = int(NDataT/batch_size)

n_iter = 300
TrainingErrorR = []
TrainingErrorS = []
ValidationErrorR = []
ValidationErrorS = []

fig, (ax1, ax2) = plt.subplots(1, 2)

# ShiftList = 20*np.array(list(range(50)))
# ShiftIntervals = [[ShiftList[i], ShiftList[i+1]] for i in range(len(ShiftList)-1)] + [[0, 200]] + [[0, 1000]] + [[0, 1000]]

# ShiftList = [0] + list(np.logspace(1, log10(Max), 10))
# ShiftList = list(map(int, ShiftList))
# ShiftIntervals = [[ShiftList[i], ShiftList[i+1]] for i in range(len(ShiftList)-1)] + [[Min, Max]]

ShiftList = list(np.logspace(1, log10(Max), 1))
ShiftList = list(map(int, ShiftList))
ShiftIntervals = [[Min, ShiftList[i]] for i in range(len(ShiftList))]

lmbd = 0.0

for ShiftInterval in ShiftIntervals:
    TrainingInput, _, TrainingRanks, TrainingSorted = MakeData(NVec=5, DVec=10, sigma=1, NData=NDataT, ShiftInterval=ShiftInterval, weights=weights, device=device)

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
            err = errR + lmbd * errS
            err.backward()
            optimizer.step()

            errorR += float(errR)/n_batch
            errorS += float(errS)/n_batch

        with torch.no_grad():
            Input = ValidationInput.to(device)
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
ax2.set_title('Erreur sorting')
ax2.plot(ValidationErrorS, 'b', label="Ensemble de Validation")
ax2.legend(loc='upper right')

plt.show()

print('end')