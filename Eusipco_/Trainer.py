from Eusipco_.DataMaker import MakeTargetedData
from Eusipco.Transformer import Network as Transformer
from Eusipco.RNN import RNNEncoder as RNN
from Eusipco.CNN import Encoder as CNN
from Complete.LRScheduler import Scheduler
from math import sqrt
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    def ChoseOutput(Pred, Input):
        Diff = Pred.unsqueeze(dim=1) - Input.unsqueeze(dim=2)
        Dist = torch.norm(Diff, dim=-1)
        Arg = torch.argmin(Dist, dim=1)
        return Arg

    import multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    ################################################################################################################################################
    # création des paramètres de la simulation

    param = {"n_encoder": 10,
             "len_in": 10,
             "d_in": 10,
             "d_att": 128,
             "network": "Transformer",
             "WidthsEmbedding": [32],
             "dropout": 0,
             "optim": "Adam",
             "lr_option": {
                 "value": 3e-4,
                 "reset": "y",
                 "type": "cos"
             },
             "mult_grad": 10000,
             "weight_decay": 1e-3,
             "NDataT": 50000,
             "NDataV": 1000,
             "batch_size": 1000,
             "n_iter": 30,
             "training_strategy": [
                 {"mean": [-10, 10], "std": [0.0001, 5]},
                 {"mean": [-10, 10], "std": [0.0001, 5]},
             ],
             "distrib": "log",
             "plot_distrib": "log",
             "error_weighting": "y",
             "max_lr": 5,
             "warmup": 2,
             "resume_from": "r",
             "period_checkpoint": 15 * 60,  # en seconde
             "nb_frames_GIF": 0,
    }

    try:
        import json
        import sys

        json_file = sys.argv[1]
        with open(json_file, "r") as f:
            temp_param = json.load(f)
        param.update(temp_param)
    except:
        print("nothing loaded")
    ################################################################################################################################################
    import os

    try:
        gpu_id = torch.cuda.current_device()
        print(f"[Worker Node {os.getenv('SLURM_NODEID')} | GPU {gpu_id}] Starting training", flush=True)
    except:
        pass
    ################################################################################################################################################
    # pour les performances
    import psutil, sys

    p = psutil.Process(os.getpid())

    if sys.platform == "win32":
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    ################################################################################################################################################
    Network = {
        "Transformer": Transformer,
        "CNN": CNN,
        "RNN": RNN
    }[param["network"]]

    period_checkpoint = param["period_checkpoint"]  # <= 0 : pas de checkpoint en entrainement, -1 : pas de sauvegarde du tout
    n_iter_window = int(param["n_iter"] / len(param["training_strategy"]))

    nb_frames_GIF = param["nb_frames_GIF"] # <= 0 : pas de GIF
    nb_frames_window = int(nb_frames_GIF / len(param["training_strategy"]))
    res_GIF = 50

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = Network(n_encoder=param["n_encoder"], d_in=param["d_in"], d_att=param["d_att"],
                WidthsEmbedding=param["WidthsEmbedding"], dropout=param["dropout"])

    N.to(device)

    optim = {
        "AdamW": torch.optim.AdamW,
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }[param['optim']]

    optimizer = optim(N.parameters(), weight_decay=param["weight_decay"], lr=param["lr_option"]["value"])

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

    if param['lr_option']['reset'] == 'y':
        n_updates = int(NDataT / batch_size) * n_iter_window
        warmup_steps = int(NDataT / batch_size * param["warmup"])
    else:
        n_updates = int(NDataT / batch_size) * n_iter
        warmup_steps = int(NDataT / batch_size * param["warmup"])

    if param['nb_frames_GIF']:
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

    ################################################################################################################################################
    from Tools.XMLTools import saveObjAsXml

    local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index("Projets"))], "Projets")
    save_dir = os.path.join(local, 'Eusipco_', 'Save')
    data_dir = os.path.join(local, 'Eusipco_', 'Data')

    try:
        from Tools.XMLTools import loadXmlAsObj

        resume_from = param["resume_from"]

        save_path = os.path.join(save_dir, resume_from)

        print(f"Reprise à partir du checkpoint : {save_path}")
        N.load_state_dict(torch.load(os.path.join(save_path, "Last_network")))
        best_state_dict = torch.load(os.path.join(save_path, "Best_network"))

        checkpoint = torch.load(os.path.join(save_path, "Scheduler.pt"))

        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

        lr_scheduler = Scheduler(optimizer=optimizer, **checkpoint["scheduler_hparams"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        error_dict = loadXmlAsObj(os.path.join(save_path, "error"))
        TrainingError = error_dict["TrainingError"]
        TrainingPerf = error_dict["TrainingPerf"]
        ValidationError = error_dict["ValidationError"]
        ValidationPerf = error_dict["ValidationPerf"]
        PlottingError = error_dict["PlottingError"]
        PlottingPerf = error_dict["PlottingPerf"]

        window_index, r = divmod(checkpoint['scheduler_state_dict']['last_epoch'] + 1, n_iter_window * n_batch * n_minibatch)
        j, r = divmod(r, n_batch * n_minibatch)
        p, k = divmod(r, n_batch)

        print(f"Reprise à la fenêtre {window_index}, itération {j}")

    except Exception as e:
        print(f"Erreur lors de la reprise du checkpoint : {e}")
        print("Lancement d'un entraînement depuis zéro.")

        if period_checkpoint != -1:
            # pour sauvegarder toutes les informations de l'apprentissage
            import datetime
            import time

            base_folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")

            attempt = 0
            while True:
                folder = f"{base_folder}({attempt})" if attempt > 0 else base_folder
                save_path = os.path.join(save_dir, param['network'] + folder)

                try:
                    os.makedirs(save_path, exist_ok=False)
                    break
                except FileExistsError:
                    attempt += 1
                    time.sleep(0.1)

            print(f"Dossier créé : {save_path}")

        lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param["max_lr"], max_steps=n_updates, type=param["lr_option"]["type"])

        TrainingError = []
        TrainingPerf = []
        ValidationError = []
        ValidationPerf = []
        PlottingError = []
        PlottingPerf = []

        window_index, j, p, k = 0, 0, 0, 0

        best_state_dict = N.state_dict().copy()

    ################################################################################################################################################

    for window in param["training_strategy"]:
        if param["lr_option"]["reset"] == "y":
            optimizer = optimizers[param['optim']](N.parameters(), weight_decay=param["weight_decay"], lr=param["lr_option"]["value"])

            n_updates = int(NDataT / batch_size) * n_iter_window
            warmup_steps = int(NDataT / batch_size * param["warmup"])
            lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param["max_lr"], max_steps=n_updates, type=param["lr_option"]["type"])

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
                torch.save(N.state_dict().copy(), os.path.join(save_path, "Last_network"))
                torch.save(Weight, os.path.join(save_path, "Weight"))

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
    torch.save(Weight, os.path.join(save_path, "Weight"))