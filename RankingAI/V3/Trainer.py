from RankingAI.V3.DataMaker import MakeData
from RankingAI.V3.Ranker import Network
from Complete.LRScheduler import Scheduler
from math import sqrt
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.cuda.set_device(2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NVec = 5
DVec = 10

N = Network(n_encoder=6, max_len=NVec, d_in=DVec, d_att=64, WidthsEmbedding=[32], n_heads=4, norm='post')

N.to(device)

optimizer = torch.optim.Adam(N.parameters(), weight_decay=1e-3, lr=3e-4)

Max = 10
Min = 0

NDataT = 500000
NDataV = 1000

weights = torch.rand(DVec, device=device)
# weights = torch.tensor([1., 0, 0, 0, 0, 0, 0, 0, 0, 0], device=device)
ValidationInput, ValidationOutput = MakeData(NVec=NVec, DVec=DVec, sigma=1, NData=NDataV, weights=weights)

batch_size = 10000
n_batch = int(NDataT/batch_size)

n_iter = 5

n_updates = int(NDataT / batch_size) * n_iter
warmup_frac = 0.2
warmup_steps = warmup_frac * n_updates
# lr_scheduler = None
lr_scheduler = Scheduler(optimizer, 64, warmup_steps, max=5)

TrainingError = []
ValidationError = []

TrainingInput, TrainingOutput = MakeData(NVec=NVec, DVec=DVec, sigma=1, NData=NDataT, weights=weights)

for j in tqdm(range(n_iter)):
    error = 0
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

        error += float(err)/n_batch

    with torch.no_grad():
        Input = ValidationInput.to(device)
        Output = ValidationOutput.to(device)
        Prediction = N(Input)

        err = torch.norm(Prediction - Output, p=2)
        ValidationError.append(float(err) / sqrt(NDataV * 5))

    TrainingError.append(error/sqrt(batch_size*5))


plt.plot(TrainingError, 'r', label="Ensemble d'entrainement")
plt.plot(ValidationError, 'b', label="Ensemble de Validation")
plt.plot([float(torch.std(ValidationOutput))]*len(TrainingError), 'black')
plt.legend(loc='upper right')
plt.title('V3')

plt.show()

print('end')