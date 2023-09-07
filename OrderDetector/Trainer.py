from OrderDetector.Detector import Network
from OrderDetector.DataMaker import MakeData
from math import sqrt
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = Network(n_encoder=6, d_in=10, d_model=64, d_att=64, num_heads=4, relative=True, masked=False)
N.to(device)


optimizer = torch.optim.Adam(N.parameters(), weight_decay=1e-3, lr=1e-3)

TrainingInput, TrainingOutput, _ = MakeData(NVec=5, DVec=10, sigma=1, NData=10000, ShiftRange=10)
ValidationInput, ValidationOutput, _ = MakeData(NVec=5, DVec=10, sigma=1, NData=1000, ShiftRange=10)

batch_size = 10000
n_batch = 1

n_iter = 1000
TrainingError = []
ValidationError = []
for j in tqdm(range(n_iter)):
    error = 0
    for i in range(n_batch):
        optimizer.zero_grad(set_to_none=True)

        InputBatch = TrainingInput[i*batch_size:(i+1)*batch_size].to(device)
        OutputBatch = TrainingOutput[i*batch_size:(i+1)*batch_size].to(device)
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
        ValidationError.append(float(err)/sqrt(1000*5))

    TrainingError.append(error/sqrt(10000*5))

plt.plot(TrainingError, 'r')
plt.plot(ValidationError, 'b')
plt.show()
print('end')