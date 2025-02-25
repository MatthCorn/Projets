from Eusipco.DataMaker import MakeTargetedData
from Eusipco.Transformer import Network as Transformer
from Eusipco.RNN import RNNEncoder as RNN
from Eusipco.CNN import Encoder as CNN
from Complete.LRScheduler import Scheduler
from math import sqrt
import torch
import numpy as np
from copy import deepcopy

def ChoseOutput(Pred, Input):
    Diff = Pred.unsqueeze(dim=1) - Input.unsqueeze(dim=2)
    Dist = torch.norm(Diff, dim=-1)
    Arg = torch.argmin(Dist, dim=1)
    return Arg

param = {"n_encoder": 10,
         "len_in": 10,
         "d_in": 10,
         "d_att": 128,
         "network": "Transformer",
         "WidthsEmbedding": [32],
         "dropout": 0,
         "lr": 3e-4,
         "mult_grad": 10000,
         "weight_decay": 1e-3,
         "NDataT": 500000,
         "NDataV": 1000,
         "batch_size": 1000,
         "n_points_reg": 10,
         "n_iter": 50,
         "training_space": {"mean": [-1000, 1000], "std": [100, 500]},
         "distrib": "log",
         "error_weighting": "y",
         "max_lr": 5,
         "warmup": 2}

################################################################################################################################################
# pour les performances
import psutil, sys, os

p = psutil.Process(os.getpid())

if sys.platform == "win32":
    p.nice(psutil.HIGH_PRIORITY_CLASS)
################################################################################################################################################


try:
    import json
    import sys
    json_file = sys.argv[1]
    with open(json_file, "r") as f:
        temp_param = json.load(f)
    param.update(temp_param)
except:
    print("nothing loaded")

Network = {"Transformer": Transformer, "CNN": CNN, "RNN": RNN}[param["network"]]

if torch.cuda.is_available():
    torch.cuda.set_device(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NDataT = param["NDataT"]
NDataV = param["NDataV"]
DVec = param["d_in"]
NVec = param["len_in"]
Weight = 2 * torch.rand(DVec) - 1
Weight = Weight / torch.norm(Weight)

mini_batch_size = 50000
n_minibatch = int(NDataT/mini_batch_size)
batch_size = param["batch_size"]
n_batch = int(mini_batch_size/batch_size)

n_iter = param["n_iter"]

n_updates = int(NDataT / batch_size) * n_iter
warmup_steps = int(NDataT / batch_size) * param["warmup"]

windows = []
if param['distrib'] == 'log':
    f = np.log
    g = np.exp
elif param['distrib'] == 'uniform':
    f = lambda x: x
    g = lambda x: x

# max_std_list = g(np.linspace(0, f(param['training_space']['std'][1]), param['n_points_reg'], endpoint=True))
# max_mean_list = 2 * max_std_list
lbd = np.flip(g(np.linspace(f(1e-4), f(1), param['n_points_reg'], endpoint=True)))

MinTrainingError = []
MaxTrainingPerf = []
MinValidationError = []
MaxValidationPerf = []

for i in range(param['n_points_reg']):
    N = Network(n_encoder=param["n_encoder"], d_in=param["d_in"], d_att=param["d_att"],
                WidthsEmbedding=param["WidthsEmbedding"], dropout=param["dropout"])
    N.to(device)

    optimizer = torch.optim.Adam(N.parameters(), weight_decay=param["weight_decay"], lr=param["lr"])

    lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param["max_lr"])

    window = deepcopy(param['training_space'])

    window['std'][0] *= lbd[i]
    # window['std'][1] = max_std_list[i]
    # window['mean'] = [-max_mean_list[i], max_mean_list[i]]


    TrainingInput, TrainingOutput, TrainingStd = MakeTargetedData(
        NVec=NVec,
        DVec=DVec,
        mean_min=window["mean"][0],
        mean_max=window["mean"][1],
        std_min=window["std"][0],
        std_max=window["std"][1],
        distrib=param["distrib"],
        NData=NDataT,
        Weight=Weight,
    )

    ValidationInput, ValidationOutput, ValidationStd = MakeTargetedData(
        NVec=NVec,
        DVec=DVec,
        mean_min=window["mean"][0],
        mean_max=window["mean"][1],
        std_min=window["std"][0],
        std_max=window["std"][1],
        distrib=param["distrib"],
        NData=NDataV,
        Weight=Weight,
    )

    TrainingError = []
    TrainingPerf = []
    ValidationError = []
    ValidationPerf = []

    for j in range(param['n_iter']):
        error = 0
        perf = 0

        for p in range(n_minibatch):
            InputMiniBatch = TrainingInput[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
            OutputMiniBatch = TrainingOutput[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
            StdMiniBatch = TrainingStd[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)

            for k in range(n_batch):
                optimizer.zero_grad(set_to_none=True)

                InputBatch = InputMiniBatch[k * batch_size:(k + 1) * batch_size]
                OutputBatch = OutputMiniBatch[k * batch_size:(k + 1) * batch_size]
                StdBatch = StdMiniBatch[k * batch_size:(k + 1) * batch_size]

                if param['error_weighting'] == 'n':
                    StdBatch = torch.mean(StdBatch)

                Prediction = N(InputBatch)

                err = torch.norm((Prediction - OutputBatch) / StdBatch, p=2) / sqrt(batch_size * DVec * NVec)
                (param["mult_grad"] * err).backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                error += float(err) / (n_batch * n_minibatch)
                perf += float(torch.sum(ChoseOutput(Prediction, InputBatch) == ChoseOutput(OutputBatch, InputBatch))) / (NDataT * NVec)

        with torch.no_grad():
            Input = ValidationInput.to(device)
            Output = ValidationOutput.to(device)
            Std = ValidationStd.to(device)

            if param['error_weighting'] == 'n':
                Std = torch.mean(Std)

            Prediction = N(Input)

            err = torch.norm((Prediction - Output) / Std, p=2) / sqrt(NDataV * DVec * NVec)
            ValidationError.append(float(err))
            ValidationPerf.append(float(torch.sum(ChoseOutput(Prediction, Input) == ChoseOutput(Output, Input))) / (NDataV * NVec))

        TrainingError.append(error)
        TrainingPerf.append(perf)

    MinTrainingError.append(min(TrainingError))
    MaxTrainingPerf.append(max(TrainingPerf))
    MinValidationError.append(min(ValidationError))
    MaxValidationPerf.append(max(ValidationPerf))

    print('############################################################')
    print('intervalle std : [', f"{window['std'][0]:.2e}", ', ', f"{window['std'][1]:.2e}", ']')
    print('intervalle mean : [', f"{window['mean'][0]:.1f}", ', ', f"{window['mean'][1]:.1f}", ']')
    print('min TrainingError : ', [f"{num:.2e}" for num in MinTrainingError])
    print('max TrainingPerf : ', [f"{num:.2e}" for num in MaxTrainingPerf])
    print('min ValidationError : ', [f"{num:.2e}" for num in MinValidationError])
    print('max ValidationPerf : ', [f"{num:.2e}" for num in MaxValidationPerf])

print('############################################################')
print('############################################################')
print('############################################################')
print('############################################################')
print('min TrainingError : ', [f"{num:.2e}" for num in MinTrainingError])
print('min TrainingPerf : ', [f"{num:.2e}" for num in MaxTrainingPerf])
print('min ValidationError : ', [f"{num:.2e}" for num in MinValidationError])
print('min ValidationPerf : ', [f"{num:.2e}" for num in MaxValidationPerf])