from RankingAI.V8.Ranker import Network
from RankingAI.V8.DataMaker import MakeData
from Complete.LRScheduler import Scheduler
from math import sqrt
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.cuda.set_device(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = Network(n_encoder=10, len_in=10, d_in=10, d_att=64, WidthsEmbedding=[32], n_heads=4, norm='post')

N.to(device)

# type_error = 'CE'
type_error = 'MSE'

optimizer = torch.optim.Adam(N.parameters(), weight_decay=1e-3, lr=3e-4)

Max = 10
Min = 0

NDataT = 50000
NDataV = 1000
DVec = 10
WeightsCut = 2 * torch.rand(DVec) - 1
WeightsSort = 2 * torch.rand(DVec) - 1
ValidationInput, ValidationOutput = MakeData(NInput=10, LimCut=0, DVec=DVec, sigma=1, NData=NDataV, WeightsCut=WeightsCut, WeightsSort=WeightsSort)

mini_batch_size = 50000
n_minibatch = int(NDataT/mini_batch_size)
batch_size = 1000
n_batch = int(mini_batch_size/batch_size)

n_iter = 3
TrainingError = []
TrainingClassError = []
ValidationError = []
ValidationClassError = []

n_updates = int(NDataT / batch_size) * n_iter
# warmup_frac = 0.2
# warmup_steps = warmup_frac * n_updates
warmup_steps = 100
lr_scheduler = None
# lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=5)

TrainingInput, TrainingOutput = MakeData(NInput=10, LimCut=0, DVec=10, sigma=1, NData=NDataT, WeightsCut=WeightsCut, WeightsSort=WeightsSort)

for j in tqdm(range(n_iter)):
    error = 0
    error_class = 0
    for p in range(n_minibatch):
        InputMiniBatch = TrainingInput[p*mini_batch_size:(p+1)*mini_batch_size].to(device)
        OutputMiniBatch = TrainingOutput[p*mini_batch_size:(p+1)*mini_batch_size].to(device)
        for k in range(n_batch):
            optimizer.zero_grad(set_to_none=True)

            InputBatch = InputMiniBatch[k*batch_size:(k+1)*batch_size]
            OutputBatch = OutputMiniBatch[k*batch_size:(k+1)*batch_size]
            Prediction = N(InputBatch)

            if type_error == 'CE':
                err = torch.norm(Prediction - OutputBatch)/sqrt(batch_size)
            if type_error == 'MSE':
                err = torch.nn.functional.mse_loss(Prediction, OutputBatch)
            err.backward()
            optimizer.step()

            ClassPrediction = F.one_hot(Prediction.argmax(dim=-1), num_classes=Prediction.size(-1))
            ErrorClassTraining = torch.norm(OutputBatch - ClassPrediction, p=2) / sqrt(2 * batch_size * 10)

            if lr_scheduler is not None:
                lr_scheduler.step()

            error += float(err)/(n_batch*n_minibatch)
            error_class += float(ErrorClassTraining)/(n_batch*n_minibatch)

    with torch.no_grad():
        Input = ValidationInput.to(device)
        Output = ValidationOutput.to(device)
        Prediction = N(Input)

        if type_error == 'CE':
            err = torch.norm(Prediction - Output)/sqrt(NDataV)
        if type_error == 'MSE':
            err = torch.nn.functional.mse_loss(Prediction, Output)

        ValidationError.append(float(err))
        ClassPrediction = F.one_hot(Prediction.argmax(dim=-1), num_classes=Prediction.size(-1))
        ErrorClassValidation = torch.norm(Output - ClassPrediction, p=2)/sqrt(2 * NDataV * 10)
        ValidationClassError.append(float(ErrorClassValidation))

    TrainingError.append(error)
    TrainingClassError.append(float(error_class))

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(TrainingError, 'r', label="Ensemble d'entrainement")
ax1.set_title('Erreur minimisée')
ax1.set_ylim(bottom=0)
ax1.plot(ValidationError, 'b', label="Ensemble de Validation")
ax1.legend(loc='upper right')

ax2.plot(TrainingClassError, 'r', label="Ensemble d'entrainement")
ax2.set_title('Erreur de classification')
ax2.set_ylim(bottom=0)
ax2.plot(ValidationClassError, 'b', label="Ensemble de Validation")
ax2.legend(loc='upper right')

fig.suptitle('V8')

plt.show()

print('end')