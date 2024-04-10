import numpy as np

from StepByStep.OrderDetector.Detector import Network
from StepByStep.OrderDetector.DataMaker import MakeData
from math import sqrt, log10
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = Network(n_encoder=5, d_in=10, d_model=64, d_att=64, WidthsEmbedding=[32], num_heads=4, relative=True, masked=False)
N.to(device)


optimizer = torch.optim.Adam(N.parameters(), weight_decay=1e-3, lr=3e-4)

Max = 10
Min = 0

NDataT = 10000
NDataV = 1000
ValidationInput, ValidationOutput, _ = MakeData(NVec=5, DVec=10, sigma=1, NData=NDataV, ShiftInterval=[Min, Max])

batch_size = 10000
n_batch = 1

n_iter = 500
TrainingError = []
ValidationError = []


ShiftList = list(np.logspace(1, log10(Max), 1))
ShiftList = list(map(int, ShiftList))
ShiftIntervals = [[Min, ShiftList[i]] for i in range(len(ShiftList))]

i = 0
for ShiftInterval in ShiftIntervals:
    TrainingInput, TrainingOutput, _ = MakeData(NVec=5, DVec=10, sigma=1, NData=NDataT, ShiftInterval=ShiftInterval)

    for j in tqdm(range(n_iter)):
        error = 0
        for k in range(n_batch):
            optimizer.zero_grad(set_to_none=True)

            InputBatch = TrainingInput[k*batch_size:(k+1)*batch_size].to(device)
            OutputBatch = TrainingOutput[k*batch_size:(k+1)*batch_size].to(device)
            Prediction = N(InputBatch)

            err = torch.norm(OutputBatch-Prediction, p=2)
            err.backward()
            optimizer.step()

            error += float(err)

        with torch.no_grad():
            InputBatch = ValidationInput.to(device)
            OutputBatch = ValidationOutput.to(device)
            Prediction = N(InputBatch)

            err = torch.norm(OutputBatch - Prediction, p=2)
            ValidationError.append(float(err)/sqrt(NDataV*5))

        TrainingError.append(error/sqrt(NDataT*5))

    Input, Output, Shift = MakeData(NVec=5, DVec=10, sigma=1, NData=10000, ShiftInterval=[Min, Max])


plt.plot(TrainingError, 'r', label="Ensemble d'entrainement")
plt.plot(ValidationError, 'b', label="Ensemble de Validation")
plt.legend(loc='upper right')

plt.show()

print('end')
