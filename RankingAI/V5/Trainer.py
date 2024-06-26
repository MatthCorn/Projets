from RankingAI.V5.Ranker import Network
from RankingAI.V5.DataMaker import MakeData
from Complete.LRScheduler import Scheduler
from math import sqrt
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.cuda.set_device(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = Network(n_encoder=8, len_in=10, len_latent=5, d_in=10, d_att=256, WidthsEmbedding=[32], n_heads=8, norm='post')

N.to(device)

optimizer = torch.optim.Adam(N.parameters(), weight_decay=1e-3, lr=3e-4)

NDataT = 5000000
NDataV = 1000
DVec = 10
WeightsCut = 2 * torch.rand(DVec) - 1
WeightsSort = 2 * torch.rand(DVec) - 1
ValidationInput, ValidationOutput = MakeData(NInput=10, NOutput=5, DVec=DVec, sigma=1, NData=NDataV, WeightsCut=WeightsCut, WeightsSort=WeightsSort)


mini_batch_size = 500000
n_minibatch = int(NDataT/mini_batch_size)
batch_size = 1000
n_batch = int(mini_batch_size/batch_size)

n_iter = 800
TrainingError = []
ValidationError = []

n_updates = int(NDataT / batch_size) * n_iter
# warmup_frac = 0.2
# warmup_steps = warmup_frac * n_updates
warmup_steps = int(NDataT / batch_size) * 100
# lr_scheduler = None
lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=5)

TrainingInput, TrainingOutput = MakeData(NInput=10, NOutput=5, DVec=10, sigma=1, NData=NDataT, WeightsCut=WeightsCut, WeightsSort=WeightsSort)

lr_list = []

for j in tqdm(range(n_iter)):
    error = 0
    lr_list.append(lr_scheduler.calc_lr(lr_scheduler._step_count))
    for p in range(n_minibatch):
        # InputMiniBatch = TrainingInput[p*mini_batch_size:(p+1)*mini_batch_size].to(device)
        # OutputMiniBatch = TrainingOutput[p*mini_batch_size:(p+1)*mini_batch_size].to(device)
        for k in range(n_batch):
            optimizer.zero_grad(set_to_none=True)

            # InputBatch = InputMiniBatch[k*batch_size:(k+1)*batch_size]
            # OutputBatch = OutputMiniBatch[k*batch_size:(k+1)*batch_size]
            # Prediction = N(InputBatch)

            # err = torch.norm(Prediction-OutputBatch, p=2)
            err = torch.tensor(0., requires_grad=True)
            err.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            error += float(err)/(n_batch*n_minibatch)

    with torch.no_grad():
        # Input = ValidationInput.to(device)
        # Output = ValidationOutput.to(device)
        # Prediction = N(Input)
        #
        # err = torch.norm(Prediction - Output, p=2)
        err = 0
        ValidationError.append(float(err)/sqrt(NDataV*5))

    TrainingError.append(error/sqrt(batch_size*5))


plt.plot(lr_list, 'r', label="Ensemble d'entrainement")
plt.ylim(bottom=0)
plt.legend(loc='upper right')
plt.title('V5')

plt.show()

print('end')