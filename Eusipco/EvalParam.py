from Eusipco.DataMaker import MakeTargetedData
from Eusipco.Transformer import Network as Transformer
from Eusipco.RNN import RNNEncoder as RNN
from Eusipco.CNN import Encoder as CNN
from Complete.LRScheduler import Scheduler
from math import sqrt
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm

################################################################################################################################################
# pour sauvegarder toutes les informations de l'apprentissage
import os
import datetime
from Tools.XMLTools import saveObjAsXml
import time

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index("Projets"))], "Projets")
base_folder = datetime.datetime.now().strftime("eval_param_%Y-%m-%d__%H-%M")
save_dir = os.path.join(local, "Eusipco", "Save")

attempt = 0
while True:
    folder = f"{base_folder}({attempt})" if attempt > 0 else base_folder
    save_path = os.path.join(save_dir, folder)

    try:
        os.makedirs(save_path, exist_ok=False)
        break
    except FileExistsError:
        attempt += 1
        time.sleep(0.1)

print(f"Dossier créé : {save_path}")
################################################################################################################################################

def eval_curve(l, n_points=5):
    n = 0
    for i in range(len(l)):
        y = l[max(0, i-n_points):min(len(l)-1, i+n_points)]
        x = np.array([i for i in range(max(0, i-n_points), min(len(l)-1, i+n_points))])
        b1 = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
        b0 = np.mean(y) - b1 * np.mean(x)

        p_y = b0 + b1 * x

        n += np.sqrt(np.mean((y - p_y)**2))

    n /= len(l)
    return p_y[-1], n


def ChoseOutput(Pred, Input):
    Diff = Pred.unsqueeze(dim=1) - Input.unsqueeze(dim=2)
    Dist = torch.norm(Diff, dim=-1)
    Arg = torch.argmin(Dist, dim=1)
    return Arg

global_param = {"n_encoder": 10,
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
         "n_iter": 80,
         "training_space": {"mean": [-10, 10], "std": [1, 5]},
         "distrib": "log",
         "error_weighting": "y",
         "max_lr": 5,
         "warmup": 2,
         "eval": {"param": "lr",
                  "n_points_reg": 5,
                  "multiplier": [1, 1e2],
                  "spacing": "log"}
         }

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
    global_param.update(temp_param)
except:
    print("nothing loaded")

Network = {"Transformer": Transformer, "CNN": CNN, "RNN": RNN}[global_param["network"]]

if torch.cuda.is_available():
    torch.cuda.set_device(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if global_param['eval']['spacing'] == 'log':
    f = np.log
    g = np.exp
elif global_param['eval']['spacing'] == 'uniform':
    f = lambda x: x
    g = lambda x: x

lbd = g(np.linspace(f(global_param['eval']['multiplier'][0]), f(global_param['eval']['multiplier'][1]), global_param['eval']['n_points_reg'], endpoint=True))

FinalPerfValidation = []
NoisePerfValidation = []
FinalErrorValidation = []
NoiseErrorValidation = []

for i in tqdm(range(global_param['eval']['n_points_reg'])):
    ValidationError = []
    ValidationPerf = []

    param = deepcopy(global_param)
    param[param['eval']['param']] = param[param['eval']['param']] * lbd[i]

    NDataT = param["NDataT"]
    NDataV = param["NDataV"]
    DVec = param["d_in"]
    NVec = param["len_in"]
    Weight = 2 * torch.rand(DVec) - 1
    Weight = Weight / torch.norm(Weight)

    mini_batch_size = 50000
    n_minibatch = int(NDataT / mini_batch_size)
    batch_size = param["batch_size"]
    n_batch = int(mini_batch_size / batch_size)

    n_iter = param["n_iter"]

    n_updates = int(NDataT / batch_size) * n_iter
    warmup_steps = int(NDataT / batch_size) * param["warmup"]

    TrainingInput, TrainingOutput, TrainingStd = MakeTargetedData(
        NVec=NVec,
        DVec=DVec,
        mean_min=param['training_space']["mean"][0],
        mean_max=param['training_space']["mean"][1],
        std_min=param['training_space']["std"][0],
        std_max=param['training_space']["std"][1],
        distrib=param["distrib"],
        NData=NDataT,
        Weight=Weight,
    )

    ValidationInput, ValidationOutput, ValidationStd = MakeTargetedData(
        NVec=NVec,
        DVec=DVec,
        mean_min=param['training_space']["mean"][0],
        mean_max=param['training_space']["mean"][1],
        std_min=param['training_space']["std"][0],
        std_max=param['training_space']["std"][1],
        distrib=param["distrib"],
        NData=NDataV,
        Weight=Weight,
    )

    N = Network(n_encoder=param["n_encoder"], d_in=param["d_in"], d_att=param["d_att"],
                WidthsEmbedding=param["WidthsEmbedding"], dropout=param["dropout"])
    N.to(device)

    optimizer = torch.optim.Adam(N.parameters(), weight_decay=param["weight_decay"], lr=param["lr"])

    lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param["max_lr"])

    for j in tqdm(range(param['n_iter'])):

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

    e, n = eval_curve(ValidationPerf)
    FinalPerfValidation.append(e)
    NoisePerfValidation.append(n)
    e, n = eval_curve(ValidationError)
    FinalErrorValidation.append(e)
    NoiseErrorValidation.append(n)

error = {"FinalPerfValidation": FinalPerfValidation,
         "NoisePerfValidation": NoisePerfValidation,
         "FinalErrorValidation": FinalErrorValidation,
         "NoiseErrorValidation": NoiseErrorValidation}

saveObjAsXml(error, os.path.join(save_path, "error"))
saveObjAsXml(global_param, os.path.join(save_path, "param"))

try:
    import os
    error_path = os.path.join(save_path, 'error')
    param_path = os.path.join(save_path, 'param')
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    from Tools.XMLTools import loadXmlAsObj
    error = loadXmlAsObj(error_path)
    param = loadXmlAsObj(param_path)
    matplotlib.use('Qt5Agg')

    if param['eval']['spacing'] == 'log':
        f = np.log
        g = np.exp
    elif param['eval']['spacing'] == 'uniform':
        f = lambda x: x
        g = lambda x: x

    lbd = g(np.linspace(f(param['eval']['multiplier'][0]), f(param['eval']['multiplier'][1]),param['eval']['n_points_reg'], endpoint=True))
    x = param[param['eval']['param']] * lbd

    upper_error = np.array(error['FinalErrorValidation']) + np.array(error['NoiseErrorValidation'])
    middle_error = np.array(error['FinalErrorValidation'])
    lower_error = np.array(error['FinalErrorValidation']) - np.array(error['NoiseErrorValidation'])

    upper_perf = np.array(error['FinalPerfValidation']) + np.array(error['NoisePerfValidation'])
    middle_perf = np.array(error['FinalPerfValidation'])
    lower_perf = np.array(error['FinalPerfValidation']) - np.array(error['NoisePerfValidation'])

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(middle_error, 'r')
    ax1.plot(upper_error, 'r')
    ax1.plot(lower_error, 'r')
    ax1.fill_between([i for i in range(len(middle_error))], lower_error, upper_error, color='r', alpha=0.5)
    ax1.set_xticks([i for i in range(len(x))])
    ax1.set_xticklabels([f"{val:.0e}" for val in x])
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel(param['eval']['param'])
    ax1.set_title("Erreur")
    ax1.set_box_aspect(1)

    ax2.plot(middle_perf, 'b')
    ax2.plot(upper_perf, 'b')
    ax2.plot(lower_perf, 'b')
    ax2.set_xticks([i for i in range(len(x))])
    ax2.fill_between([i for i in range(len(middle_perf))], upper_perf, lower_perf, color='b', alpha=0.5)
    ax2.set_ylim(bottom=0)
    ax2.set_xticklabels([f"{val:.0e}" for val in x])
    ax2.set_xlabel(param['eval']['param'])
    ax2.set_title("Accuracy")
    ax2.set_box_aspect(1)

    fig.tight_layout(pad=1.0)

    plt.show()
except:
    pass