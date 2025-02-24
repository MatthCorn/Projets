from Eusipco.DataMaker import MakeTargetedData
from Eusipco.Transformer import Network as Transformer
from Eusipco.RNN import RNNEncoder as RNN
from Eusipco.CNN import Encoder as CNN
from Complete.LRScheduler import Scheduler
from GradObserver.GradObserverClass import DictGradObserver
from Tools.ParamObs import DictParamObserver
from Tools.GitPush import git_push
from math import sqrt
import torch
from tqdm import tqdm

def ChoseOutput(Pred, Input):
    Diff = Pred.unsqueeze(dim=1) - Input.unsqueeze(dim=2)
    Dist = torch.norm(Diff, dim=-1)
    Arg = torch.argmin(Dist, dim=1)
    return Arg

################################################################################################################################################
# pour sauvegarder toutes les informations de l"apprentissage
import os
import datetime
from Tools.XMLTools import saveObjAsXml
import pickle

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index("Projets"))], "Projets")
folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
save_path = os.path.join(local, "Eusipco", "Save", folder)
################################################################################################################################################

################################################################################################################################################
# pour les performances
import psutil, sys

p = psutil.Process(os.getpid())

if sys.platform == "win32":
    p.nice(psutil.HIGH_PRIORITY_CLASS)
################################################################################################################################################

param = {"n_encoder": 10,
         "len_in": 10,
         "d_in": 10,
         "d_att": 128,
         "network": "Transformer",
         "WidthsEmbedding": [32],
         "dropout": 0,
         "optim": "Adam",
         "lr": 3e-4,
         "mult_grad": 10000,
         "weight_decay": 1e-3,
         "NDataT": 500000,
         "NDataV": 1000,
         "batch_size": 1000,
         "n_iter": 800,
         "training_strategy": [
             {"mean": [-1000, 1000], "std": [0.1, 500]}
         ],
         "distrib": "log",
         "plot_distrib": "log",
         "max_lr": 5,
         "FreqGradObs": 1/3,
         "warmup": 2}

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

freq_checkpoint = 1/10
nb_frames_GIF = 100
nb_frames_window = int(nb_frames_GIF / len(param["training_strategy"]))
res_GIF = 50
n_iter_window = int(param["n_iter"] / len(param["training_strategy"]))

if torch.cuda.is_available():
    torch.cuda.set_device(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = Network(n_encoder=param["n_encoder"], d_in=param["d_in"], d_att=param["d_att"],
            WidthsEmbedding=param["WidthsEmbedding"], dropout=param["dropout"])

DictGrad = DictGradObserver(N)

N.to(device)

optimizers = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}

optimizer = optimizers[param['optim']](N.parameters(), weight_decay=param["weight_decay"], lr=param["lr"])

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
TrainingError = []
TrainingPerf = []
ValidationError = []
ValidationPerf = []
PlottingError = []
PlottingPerf = []

n_updates = int(NDataT / batch_size) * n_iter
warmup_steps = int(NDataT / batch_size * param["warmup"])
lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param["max_lr"])

PlottingInput, PlottingOutput, PlottingStd = MakeTargetedData(
    NVec=NVec,
    DVec=DVec,
    mean_min=min([window["mean"][0] for window in param["training_strategy"]]),
    mean_max=max([window["mean"][1] for window in param["training_strategy"]]),
    std_min=min([window["std"][0] for window in param["training_strategy"]]),
    std_max=max([window["std"][1] for window in param["training_strategy"]]),
    distrib=param["plot_distrib"],
    NData=res_GIF,
    Weight=Weight,
    plot=True,
)

best_state_dict = N.state_dict().copy()

for window in param["training_strategy"]:
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

    for j in tqdm(range(n_iter_window)):
        error = 0
        perf = 0
        time_to_observe = (int(j * param["FreqGradObs"]) == (j * param["FreqGradObs"]))
        time_for_checkpoint = (int(j * freq_checkpoint) == (j * freq_checkpoint))
        time_for_GIF = (j in torch.linspace(0, n_iter_window, nb_frames_window, dtype=int))

        for p in range(n_minibatch):
            InputMiniBatch = TrainingInput[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
            OutputMiniBatch = TrainingOutput[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
            StdMiniBatch = TrainingStd[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)

            for k in range(n_batch):
                optimizer.zero_grad(set_to_none=True)

                InputBatch = InputMiniBatch[k * batch_size:(k + 1) * batch_size]
                OutputBatch = OutputMiniBatch[k * batch_size:(k + 1) * batch_size]
                StdBatch = StdMiniBatch[k * batch_size:(k + 1) * batch_size]
                Prediction = N(InputBatch)

                err = torch.norm((Prediction - OutputBatch) / StdBatch, p=2) / sqrt(batch_size * DVec * NVec)
                (param["mult_grad"] * err).backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                if k == 0 and time_to_observe:
                    DictGrad.update()

                error += float(err) / (n_batch * n_minibatch)
                perf += float(torch.sum(ChoseOutput(Prediction, InputBatch) == ChoseOutput(OutputBatch, InputBatch))) / (NDataT * NVec)

        if time_to_observe:
            DictGrad.next(j)

        with torch.no_grad():
            Input = ValidationInput.to(device)
            Output = ValidationOutput.to(device)
            Std = ValidationStd.to(device)
            Prediction = N(Input)

            err = torch.norm((Prediction - Output) / Std, p=2) / sqrt(NDataV * DVec * NVec)
            ValidationError.append(float(err))
            ValidationPerf.append(float(torch.sum(ChoseOutput(Prediction, Input) == ChoseOutput(Output, Input))) / (NDataV * NVec))

        TrainingError.append(error)
        TrainingPerf.append(perf)

        if error == min(TrainingError):
            best_state_dict = N.state_dict().copy()

        if time_for_GIF:
            with torch.no_grad():
                Input = PlottingInput.to(device)
                Output = PlottingOutput.to(device)
                Std = PlottingStd.to(device)
                Prediction = N(Input)

                err = torch.norm((Prediction - Output) / Std, p=2, dim=[-1, -2]) / sqrt(DVec * NVec)
                perf = torch.sum(ChoseOutput(Prediction, Input) == ChoseOutput(Output, Input), dim=[-1]) / NVec
                PlottingError.append(err.reshape(res_GIF, res_GIF).tolist())
                PlottingPerf.append(perf.reshape(res_GIF, res_GIF).tolist())

        if time_for_checkpoint:
            try:
                os.mkdir(save_path)
            except:
                pass
            error = {"TrainingError": TrainingError,
                     "ValidationError": ValidationError,
                     "TrainingPerf": TrainingPerf,
                     "ValidationPerf": ValidationPerf,
                     "PlottingPerf": PlottingPerf,
                     "PlottingError": PlottingError}
            saveObjAsXml(param, os.path.join(save_path, "param"))
            saveObjAsXml(error, os.path.join(save_path, "error"))
            torch.save(best_state_dict, os.path.join(save_path, "Best_network"))
            with open(os.path.join(save_path, "DictGrad.pkl"), "wb") as file:
                pickle.dump(DictGrad, file)
            with open(os.path.join(save_path, "ParamObs.pkl"), "wb") as file:
                ParamObs = DictParamObserver(N)
                pickle.dump(ParamObs, file)

git_push(local, save_path, CommitMsg='simu ' + folder)