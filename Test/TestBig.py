from RankAI.V4.Vecteurs.DataMaker import MakeTargetedData
from RankAI.V4.Vecteurs.Ranker import Network, ChoseOutput
from Complete.LRScheduler import Scheduler
from GradObserver.GradObserverClass import DictGradObserver
from Tools.ParamObs import DictParamObserver
from math import sqrt
import torch
from tqdm import tqdm

################################################################################################################################################
# pour sauvegarder toutes les informations de l'apprentissage
import os
import datetime
from Tools.XMLTools import saveObjAsXml
import pickle
import time

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index("Projets"))], "Projets")
base_folder = datetime.datetime.now().strftime("Test_big_%Y-%m-%d__%H-%M")
save_dir = os.path.join(local, 'RankAI', 'Save', 'V4', 'Vecteurs')

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

################################################################################################################################################
# pour les performances
import psutil, sys

p = psutil.Process(os.getpid())

if sys.platform == "win32":
    p.nice(psutil.HIGH_PRIORITY_CLASS)
################################################################################################################################################

param = {"n_encoder": 6,
         "len_in": 10,
         'len_out': 5,
         "d_in": 10,
         "d_att": 512,
         "WidthsEmbedding": [64],
         "width_FF": [2048],
         'n_heads': 8,
         "dropout": 0,
         'norm': 'post',
         "optim": "Adam",
         "lr_option": {
             "value": 1e-5,
             "reset": "y",
             "type": "cos"
         },
         "mult_grad": 10000,
         "weight_decay": 1e-3,
         "NDataT": 5000000,
         "NDataV": 1000,
         "batch_size": 200,
         "n_iter": 40,
         "training_strategy": [
             {"mean": [-5, 5], "std": [1, 2]},
         ],
         "distrib": "log",
         "plot_distrib": "log",
         "error_weighting": "y",
         "max_lr": 5,
         "FreqGradObs": 1/100,
         "warmup": 5}

try:
    import json
    import sys
    json_file = sys.argv[1]
    with open(json_file, "r") as f:
        temp_param = json.load(f)
    param.update(temp_param)
except:
    print("nothing loaded")

freq_checkpoint = 1/3
nb_frames_GIF = 100
nb_frames_window = int(nb_frames_GIF / len(param["training_strategy"]))
res_GIF = 50
n_iter_window = int(param["n_iter"] / len(param["training_strategy"]))

if torch.cuda.is_available():
    torch.cuda.set_device(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = Network(n_encoder=param['n_encoder'], len_in=param['len_in'], len_latent=param['len_out'], d_in=param['d_in'],
            d_att=param['d_att'], WidthsEmbedding=param['WidthsEmbedding'], width_FF=param['width_FF'],
            n_heads=param['n_heads'], norm=param['norm'], dropout=param['dropout'])

DictGrad = DictGradObserver(N)

N.to(device)

optimizers = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}

optimizer = optimizers[param['optim']](N.parameters(), weight_decay=param["weight_decay"], lr=param["lr_option"]["value"])

NDataT = param['NDataT']
NDataV = param['NDataV']
DVec = param['d_in']
NInput = param['len_in']
NOutput = param['len_out']
WeightN = 2 * torch.rand(DVec) - 1
WeightN = WeightN / torch.norm(WeightN)
WeightF = 2 * torch.rand(DVec) - 1
WeightF = WeightF / torch.norm(WeightF)

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
lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param["max_lr"], max_steps=n_updates, type=param["lr_option"]["type"])

PlottingInput, PlottingOutput, PlottingStd = MakeTargetedData(
    NInput=NInput,
    NOutput=NOutput,
    DVec=DVec,
    mean_min=min([window["mean"][0] for window in param["training_strategy"]]),
    mean_max=max([window["mean"][1] for window in param["training_strategy"]]),
    std_min=min([window["std"][0] for window in param["training_strategy"]]),
    std_max=max([window["std"][1] for window in param["training_strategy"]]),
    distrib=param["plot_distrib"],
    NData=res_GIF,
    WeightF=WeightF,
    WeightN=WeightN,
    plot=True,
)

best_state_dict = N.state_dict().copy()

for window in param["training_strategy"]:
    if param["lr_option"]["reset"] == "y":
        optimizer = optimizers[param['optim']](N.parameters(), weight_decay=param["weight_decay"], lr=param["lr_option"]["value"])

        n_updates = int(NDataT / batch_size) * n_iter_window
        warmup_steps = int(NDataT / batch_size * param["warmup"])
        lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param["max_lr"], max_steps=n_updates, type=param["lr_option"]["type"])

    TrainingInput, TrainingOutput, TrainingStd = MakeTargetedData(
        NInput=NInput,
        NOutput=NOutput,
        DVec=DVec,
        mean_min=window["mean"][0],
        mean_max=window["mean"][1],
        std_min=window["std"][0],
        std_max=window["std"][1],
        distrib=param["distrib"],
        NData=NDataT,
        WeightF=WeightF,
        WeightN=WeightN,
    )

    ValidationInput, ValidationOutput, ValidationStd = MakeTargetedData(
        NInput=NInput,
        NOutput=NOutput,
        DVec=DVec,
        mean_min=window["mean"][0],
        mean_max=window["mean"][1],
        std_min=window["std"][0],
        std_max=window["std"][1],
        distrib=param["distrib"],
        NData=NDataV,
        WeightF=WeightF,
        WeightN=WeightN,
    )

    for j in tqdm(range(n_iter_window)):
        error = 0
        perf = 0
        time_to_observe = (int(j * param["FreqGradObs"]) == (j * param["FreqGradObs"]))
        time_for_checkpoint = (int(j * freq_checkpoint) == (j * freq_checkpoint))
        time_for_GIF = (j in torch.linspace(0, n_iter_window, nb_frames_window, dtype=int))

        for p in tqdm(range(n_minibatch)):
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

                err = torch.norm((Prediction - OutputBatch) / StdBatch, p=2) / sqrt(batch_size * DVec * NOutput)
                (param["mult_grad"] * err).backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                if k == 0 and time_to_observe:
                    DictGrad.update()

                TrainingError.append(float(err))
                TrainingPerf.append(float(torch.sum(ChoseOutput(Prediction, InputBatch) == ChoseOutput(OutputBatch, InputBatch))) / (batch_size * NOutput))

                error += float(err) / (n_batch * n_minibatch)
                perf += float(torch.sum(ChoseOutput(Prediction, InputBatch) == ChoseOutput(OutputBatch, InputBatch))) / (NDataT * NOutput)

        if time_to_observe:
            DictGrad.next(j)

        with torch.no_grad():
            Input = ValidationInput.to(device)
            Output = ValidationOutput.to(device)
            Std = ValidationStd.to(device)

            if param['error_weighting'] == 'n':
                Std = torch.mean(Std)

            Prediction = N(Input)

            err = torch.norm((Prediction - Output) / Std, p=2) / sqrt(NDataV * DVec * NOutput)
            ValidationError.append(float(err))
            ValidationPerf.append(float(torch.sum(ChoseOutput(Prediction, Input) == ChoseOutput(Output, Input))) / (NDataV * NOutput))

        # TrainingError.append(error)
        # TrainingPerf.append(perf)

        if error == min(TrainingError):
            best_state_dict = N.state_dict().copy()

        if time_for_GIF:
            with torch.no_grad():
                Input = PlottingInput.to(device)
                Output = PlottingOutput.to(device)
                Std = PlottingStd.to(device)

                if param['error_weighting'] == 'n':
                    Std = torch.mean(Std)

                Prediction = N(Input)

                err = torch.norm((Prediction - Output) / Std, p=2, dim=[-1, -2]) / sqrt(DVec * NOutput)
                perf = torch.sum(ChoseOutput(Prediction, Input) == ChoseOutput(Output, Input), dim=[-1]) / NOutput
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
            torch.save(N.state_dict().copy(), os.path.join(save_path, "Last_network"))
            torch.save(WeightN, os.path.join(save_path, "WeightN"))
            torch.save(WeightF, os.path.join(save_path, "WeightF"))
            with open(os.path.join(save_path, "DictGrad.pkl"), "wb") as file:
                pickle.dump(DictGrad, file)
            with open(os.path.join(save_path, "ParamObs.pkl"), "wb") as file:
                ParamObs = DictParamObserver(N)
                pickle.dump(ParamObs, file)

error = {"TrainingError": TrainingError,
         "ValidationError": ValidationError,
         "TrainingPerf": TrainingPerf,
         "ValidationPerf": ValidationPerf,
         "PlottingPerf": PlottingPerf,
         "PlottingError": PlottingError}
saveObjAsXml(param, os.path.join(save_path, "param"))
saveObjAsXml(error, os.path.join(save_path, "error"))
torch.save(best_state_dict, os.path.join(save_path, "Best_network"))
torch.save(N.state_dict().copy(), os.path.join(save_path, "Last_network"))
torch.save(WeightN, os.path.join(save_path, "WeightN"))
torch.save(WeightF, os.path.join(save_path, "WeightF"))
with open(os.path.join(save_path, "DictGrad.pkl"), "wb") as file:
    pickle.dump(DictGrad, file)
with open(os.path.join(save_path, "ParamObs.pkl"), "wb") as file:
    ParamObs = DictParamObserver(N)
    pickle.dump(ParamObs, file)