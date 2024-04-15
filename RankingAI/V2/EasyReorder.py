import numpy as np
from RankingAI.V2.Ranker import Network

from RankingAI.V2.DataMaker import CheatedData
from math import sqrt, log10
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


N = Network(d_in=10, d_att=64, n_heads=4)
N.to(device)


optimizer = torch.optim.Adam(N.parameters(), weight_decay=1e-3, lr=3e-4)

Max = 10
Min = 0

NDataT = 100000
NDataV = 1000
DVec = 10
weights = torch.rand(DVec, device=device)
# weights = torch.tensor([1., 0, 0, 0, 0, 0, 0, 0, 0, 0], device=device)
InputV, InitialPEV, FinalPEV, OutputV, RanksV = CheatedData(NVec=5, DVec=DVec, sigma=1, NData=NDataV, ShiftInterval=[Min, Max], d_att=64, weights=weights)

batch_size = 10000
n_batch = int(NDataT/batch_size)

lmd = 0

n_iter = 50
TrainingErrorR = []
TrainingErrorS = []
ValidationErrorR = []
ValidationErrorS = []

ShiftList = list(np.logspace(1, log10(Max), 1))
ShiftList = list(map(int, ShiftList))
ShiftIntervals = [[Min, ShiftList[i]] for i in range(len(ShiftList))]

i = 0
for ShiftInterval in ShiftIntervals:
    InputT, InitialPET, FinalPET, OutputT, RanksT = CheatedData(NVec=5, DVec=10, sigma=1, NData=NDataT, d_att=64, ShiftInterval=ShiftInterval, weights=weights)

    for j in tqdm(range(n_iter)):
        errorR = 0
        errorS = 0
        for k in range(n_batch):
            optimizer.zero_grad(set_to_none=True)

            InputBatch = InputT[k*batch_size:(k+1)*batch_size].to(device)
            InitialBatch = InitialPET[k*batch_size:(k+1)*batch_size].to(device)
            FinalBatch = FinalPET[k*batch_size:(k+1)*batch_size].to(device)

            OutputBatch = OutputT[k*batch_size:(k+1)*batch_size].to(device)
            RanksBatch = RanksT[k*batch_size:(k+1)*batch_size].to(device)
            Prediction, Ranks = N(InputBatch, InitialBatch, FinalBatch)

            errS = torch.norm(OutputBatch-Prediction, p=2)
            errR = torch.norm(RanksBatch - Ranks, p=2)
            err = lmd * errS + (1 - lmd) * errR
            err.backward()
            optimizer.step()

            errorR += float(errR)/n_batch
            errorS += float(errS)/n_batch

        with torch.no_grad():
            InputV = InputV.to(device)
            InitialV = InitialPEV.to(device)
            FinalV = FinalPEV.to(device)

            OutputV = OutputV.to(device)
            RanksV = RanksV.to(device)
            Prediction, Ranks = N(InputV, InitialV, FinalV)

            errS = torch.norm(OutputV - Prediction, p=2)
            errR = torch.norm(RanksV - Ranks, p=2)
            ValidationErrorR.append(float(errR)/sqrt(NDataV*5))
            ValidationErrorS.append(float(errS)/sqrt(NDataV*5))

        TrainingErrorR.append(errorR/sqrt(batch_size*5))
        TrainingErrorS.append(errorS/sqrt(batch_size*5))

print(torch.std(OutputV))

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(TrainingErrorS, 'r', label="Ensemble d'entrainement")
ax1.plot(ValidationErrorS, 'b', label="Ensemble de Validation")
ax1.legend(loc='upper right')
ax1.set_title('trie')

ax2.plot(TrainingErrorR, 'r', label="Ensemble d'entrainement")
ax2.plot(ValidationErrorR, 'b', label="Ensemble de Validation")
ax2.legend(loc='upper right')
ax2.set_title('position')

plt.show()

print('end')