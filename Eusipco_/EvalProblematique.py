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
base_folder = datetime.datetime.now().strftime("eval_problem_%Y-%m-%d__%H-%M")
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
         "lr": 3e-3,
         "mult_grad": 10000,
         "weight_decay": 1e-3,
         "NDataT": 50000,
         "NDataV": 1000,
         "batch_size": 1000,
         "n_points_reg": 10,
         "n_iter": 5,
         "training_space": {"mean": [-10, 10], "std": [1, 5]},
         "lbd": {"min": 1e-4, "max": 1},
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

lbd = np.flip(g(np.linspace(f(param['lbd']['min']), f(param['lbd']['max']), param['n_points_reg'], endpoint=True)))

ErrorQ1List = []
ErrorQ2List = []
ErrorQ3List = []
Error_lower_whiskerList = []
Error_upper_whiskerList = []
ErrorOutliersList = []

PerfQ1List = []
PerfQ2List = []
PerfQ3List = []
Perf_lower_whiskerList = []
Perf_upper_whiskerList = []
PerfOutliersList = []

for i in tqdm(range(param['n_points_reg'])):
    N = Network(n_encoder=param["n_encoder"], d_in=param["d_in"], d_att=param["d_att"],
                WidthsEmbedding=param["WidthsEmbedding"], dropout=param["dropout"])
    N.to(device)

    optimizer = torch.optim.Adam(N.parameters(), weight_decay=param["weight_decay"], lr=param["lr"])

    lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param["max_lr"])

    window = deepcopy(param['training_space'])

    window['std'][0] *= lbd[i]

    Error = []
    ErrorQ1 = []
    ErrorQ2 = []
    ErrorQ3 = []
    Error_lower_whisker = []
    Error_upper_whisker = []
    ErrorOutliers = []

    PerfQ1 = []
    PerfQ2 = []
    PerfQ3 = []
    Perf_lower_whisker = []
    Perf_upper_whisker = []
    PerfOutliers = []

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

            err = torch.norm((Prediction - Output) / Std, p=2, dim=[1, 2]) / sqrt(DVec * NVec)
            Error.append(torch.mean(err))

            data = err.cpu().numpy()
            ErrorQ1.append(np.percentile(data, 25))
            ErrorQ2.append(np.median(data))
            ErrorQ3.append(np.percentile(data, 75))
            IQR = ErrorQ3[-1] - ErrorQ1[-1]

            Error_lower_whisker.append(max(ErrorQ1[-1] - 1.5 * IQR, min(data)))
            Error_upper_whisker.append(min(ErrorQ3[-1] + 1.5 * IQR, max(data)))

            ErrorOutliers.append(data[(data < Error_lower_whisker[-1]) | (data > Error_upper_whisker[-1])])

            perf = torch.mean((ChoseOutput(Prediction, Input) == ChoseOutput(Output, Input)).to(float), dim=-1)

            data = perf.cpu().numpy()
            PerfQ1.append(np.percentile(data, 25))
            PerfQ2.append(np.median(data))
            PerfQ3.append(np.percentile(data, 75))
            IQR = PerfQ3[-1] - PerfQ1[-1]

            Perf_lower_whisker.append(max(PerfQ1[-1] - 1.5 * IQR, min(data)))
            Perf_upper_whisker.append(min(PerfQ3[-1] + 1.5 * IQR, max(data)))

            PerfOutliers.append(data[(data < Perf_lower_whisker[-1]) | (data > Perf_upper_whisker[-1])])


    argmin = Error.index(min(Error))
    ErrorQ1List.append(ErrorQ1[argmin])
    ErrorQ2List.append(ErrorQ2[argmin])
    ErrorQ3List.append(ErrorQ3[argmin])

    Error_lower_whiskerList.append(Error_lower_whisker[argmin])
    Error_upper_whiskerList.append(Error_upper_whisker[argmin])

    ErrorOutliersList.append(ErrorOutliers[argmin])

    PerfQ1List.append(PerfQ1[argmin])
    PerfQ2List.append(PerfQ2[argmin])
    PerfQ3List.append(PerfQ3[argmin])

    Perf_lower_whiskerList.append(Perf_lower_whisker[argmin])
    Perf_upper_whiskerList.append(Perf_upper_whisker[argmin])

    PerfOutliersList.append(PerfOutliers[argmin])


error = {"ErrorQ1List": ErrorQ1List,
         "ErrorQ2List": ErrorQ2List,
         "ErrorQ3List": ErrorQ3List,
         "Error_lower_whiskerList": Error_lower_whiskerList,
         "Error_upper_whiskerList": Error_upper_whiskerList,
         "ErrorOutliersList": ErrorOutliersList,
         "PerfQ1List": PerfQ1List,
         "PerfQ2List": PerfQ2List,
         "PerfQ3List": PerfQ3List,
         "Perf_lower_whiskerList": Perf_lower_whiskerList,
         "Perf_upper_whiskerList": Perf_upper_whiskerList,
         "PerfOutliersList": PerfOutliersList,}

saveObjAsXml(error, os.path.join(save_path, "error"))
saveObjAsXml(param, os.path.join(save_path, "param"))

try:
    import os
    error_path = os.path.join(save_path, 'error')
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    from Tools.XMLTools import loadXmlAsObj
    error = loadXmlAsObj(error_path)
    matplotlib.use('Qt5Agg')

    upper_error = np.array(error['MinError']) + np.array(error['RightStdMinError'])
    middle_error = np.array(error['MinError'])
    lower_error = np.array(error['MinError']) - np.array(error['LeftStdMinError'])

    upper_perf = np.array(error['MaxPerf']) + np.array(error['RightStdMaxPerf'])
    middle_perf = np.array(error['MaxPerf'])
    lower_perf = np.array(error['MaxPerf']) - np.array(error['LeftStdMaxPerf'])

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(middle_error, 'r')
    ax1.plot(upper_error, 'lightcoral')
    ax1.plot(lower_error, 'lightcoral')
    ax1.set_ylim(bottom=0)
    ax1.set_title("Erreur")
    ax1.set_box_aspect(1)

    ax2.plot(middle_perf, 'b')
    ax2.plot(upper_perf, 'skyblue')
    ax2.plot(lower_perf, 'skyblue')
    ax2.set_ylim(bottom=0)
    ax2.set_title("Accuracy")
    ax2.set_box_aspect(1)

    fig.tight_layout(pad=1.0)

    plt.show()
except:
    pass