import numpy as np

from OrderDetector.Detector import Network
from OrderDetector.DataMaker import MakeData
from math import sqrt, log10
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = Network(n_encoder=5, d_in=10, d_model=64, d_att=64, WidthsEmbedding=[32], num_heads=4, relative=True, masked=False)
N.to(device)


optimizer = torch.optim.Adam(N.parameters(), weight_decay=1e-3, lr=3e-4)

Max = 10000
Min = 0

NDataT = 10000
NDataV = 1000
ValidationInput, ValidationOutput, _ = MakeData(NVec=5, DVec=10, sigma=1, NData=NDataV, ShiftInterval=[Min, Max])

batch_size = 10000
n_batch = 1

n_iter = 1000
TrainingError = []
ValidationError = []

fig, (ax1, ax2) = plt.subplots(1, 2)

# ShiftList = 20*np.array(list(range(50)))
# ShiftIntervals = [[ShiftList[i], ShiftList[i+1]] for i in range(len(ShiftList)-1)] + [[0, 200]] + [[0, 1000]] + [[0, 1000]]


ShiftList = [0] + list(np.logspace(1, log10(Max), 10))
ShiftList = list(map(int, ShiftList))
ShiftIntervals = [[ShiftList[i], ShiftList[i+1]] for i in range(len(ShiftList)-1)] + [[Min, Max]]


for ShiftInterval in ShiftIntervals:
    TrainingInput, TrainingOutput, _ = MakeData(NVec=5, DVec=10, sigma=1, NData=NDataT, ShiftInterval=ShiftInterval)

    if ShiftInterval == [Min, Max]:
        n_iter = 3000

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

    with torch.no_grad():
        Input = Input.to(device)
        Output = Output.to(device)
        Prediction = N(Input)
        ErrShift = torch.norm(Output - Prediction, dim=(1, 2)).cpu().numpy() / sqrt(5)
    Shift = Shift[:, 0, 0].numpy()

    #g, f, \
    e, d, c, b, a = np.polyfit(Shift, ErrShift, deg=4)

    x = np.linspace(0, ShiftList[-1], 100)
    y = a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 #+ f * x ** 5 + g * x ** 6
    ax2.plot(x, y, label=str(ShiftInterval[1]))



ax1.plot(TrainingError, 'r', label="Ensemble d'entrainement")
ax1.set_title('Erreur gobale')
ax1.plot(ValidationError, 'b', label="Ensemble de Validation")
ax1.legend(loc='upper right')

ax2.set_title('Erreur Shift')
ax2.legend(loc='upper right')

plt.show()

print('end')