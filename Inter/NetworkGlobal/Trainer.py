from Inter.Model.DataMaker import GetData
from Inter.NetworkGlobal.Network import TransformerTranslator
from Complete.LRScheduler import Scheduler
from GradObserver.GradObserverClass import DictGradObserver
from Tools.ParamObs import DictParamObserver
from math import sqrt
import torch
from tqdm import tqdm
import time

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    ################################################################################################################################################
    # création des paramètres de la simulation

    param = {"n_encoder": 10,
             "n_decoder": 10,
             "len_in": 10,
             "len_out": 20,
             "n_pulse_plateau": 6,
             "n_sat": 5,
             "n_mes": 6,
             "sensitivity": 0.1,
             "d_in": 10,
             "d_att": 128,
             "widths_embedding": [32],
             'width_FF': [256],
             'n_heads': 4,
             "dropout": 0,
             'norm': 'post',
             "optim": "Adam",
             "lr_option": {
                 "value": 1e-4,
                 "reset": "y",
                 "type": "cos"
             },
             "mult_grad": 10000,
             "weight_decay": 1e-3,
             "NDataT": 50000,
             "NDataV": 1000,
             "batch_size": 1000,
             "n_iter": 20,
             "training_strategy": [
                 {"mean": [-5, 5], "std": [0.2, 1]},
             ],
             "distrib": "log",
             "plot_distrib": "log",
             "error_weighting": "y",
             "max_lr": 5,
             "FreqGradObs": 1/3,
             "warmup": 5,
             "resume_from": "r",
             "period_checkpoint": 15 * 60,  # en seconde
             "nb_frames_GIF": -1
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
    import psutil, sys, os

    p = psutil.Process(os.getpid())

    if sys.platform == "win32":
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    ################################################################################################################################################

    d_out = param['d_in'] + 1

    period_checkpoint = param["period_checkpoint"]  #  <= 0 : pas de checkpoint en entrainement, -1 : pas de sauvegarde du tout
    nb_frames_GIF = param["nb_frames_GIF"]
    nb_frames_window = int(nb_frames_GIF / len(param["training_strategy"]))
    res_GIF = 50
    n_iter_window = int(param["n_iter"] / len(param["training_strategy"]))

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = TransformerTranslator(param['d_in'], d_out, d_att=param['d_att'], n_heads=param['n_heads'], n_encoders=param['n_encoder'],
                              n_decoders=param['n_decoder'], widths_embedding=param['widths_embedding'], len_in=param['len_in'],
                              len_out=param['len_out'], norm=param['norm'], dropout=param['dropout'], width_FF=param['width_FF'])

    N.to(device)

    NDataT = param['NDataT']
    NDataV = param['NDataV']
    DInput = param['d_in']
    NInput = param['len_in']
    NOutput = param['len_out']
    weight_f = torch.tensor([1., 0.] + [0.] * (param['d_in'] - 3)).numpy()
    weight_l = torch.tensor([0., 1.] + [0.] * (param['d_in'] - 3)).numpy()

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

    if nb_frames_GIF > 0:
        PlottingInput, PlottingOutput, PlottingMasks, PlottingStd = GetData(
            d_in=param['d_in'],
            n_pulse_plateau=param["n_pulse_plateau"],
            n_sat=param["n_sat"],
            n_mes=param['n_mes'],
            len_in=param["len_in"],
            len_out=param["len_out"],
            n_data_training=res_GIF,
            sensitivity=param["sensitivity"],
            bias='freq',
            mean_min=min([window["mean"][0] for window in param["training_strategy"]]),
            mean_max=max([window["mean"][1] for window in param["training_strategy"]]),
            std_min=min([window["std"][0] for window in param["training_strategy"]]),
            std_max=max([window["std"][1] for window in param["training_strategy"]]),
            distrib=param["plot_distrib"],
            weight_f=weight_f,
            weight_l=weight_l,
            plot=True,
            type='complete',
            parallel=True
        )

    optimizers = {
        "AdamW": torch.optim.AdamW,
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }

    ################################################################################################################################################
    import pickle
    from Tools.XMLTools import saveObjAsXml
    local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index("Projets"))], "Projets")
    save_dir = os.path.join(local, 'Inter', 'NetworkGlobal', 'Save')
    data_dir = os.path.join(local, 'Inter', 'Data')

    try:
        from Tools.XMLTools import loadXmlAsObj
        resume_from = param["resume_from"]

        save_path = os.path.join(save_dir, resume_from)

        print(f"Reprise à partir du checkpoint : {save_path}")
        N.load_state_dict(torch.load(os.path.join(save_path, "Last_network")))
        with open(os.path.join(save_path, "DictGrad.pkl"), "rb") as f:
            DictGrad = pickle.load(f)
        DictGrad.reconnect(N)

        checkpoint = torch.load(os.path.join(save_path, "Scheduler.pt"))

        optimizer = optimizers[param['optim']](N.parameters(), weight_decay=param["weight_decay"],
                                               lr=param["lr_option"]["value"])

        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

        lr_scheduler = Scheduler(optimizer=optimizer, **checkpoint["scheduler_hparams"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        error_dict = loadXmlAsObj(os.path.join(save_path, "error"))
        TrainingError = error_dict["TrainingError"]
        ValidationError = error_dict["ValidationError"]
        PlottingError = error_dict["PlottingError"]

        window_index, r = divmod(checkpoint['scheduler_state_dict']['last_epoch'] + 1,
                                 n_iter_window * n_batch * n_minibatch)
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
                save_path = os.path.join(save_dir, folder)

                try:
                    os.makedirs(save_path, exist_ok=False)
                    break
                except FileExistsError:
                    attempt += 1
                    time.sleep(0.1)

            print(f"Dossier créé : {save_path}")

        optimizer = optimizers[param['optim']](N.parameters(), weight_decay=param["weight_decay"],
                                               lr=param["lr_option"]["value"])
        lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param["max_lr"], max_steps=n_updates,
                                 type=param["lr_option"]["type"])

        DictGrad = DictGradObserver(N)

        TrainingError = []
        ValidationError = []
        PlottingError = []

        window_index ,j ,p, k = 0, 0, 0, 0

        best_state_dict = N.state_dict().copy()

    ################################################################################################################################################

    while window_index < len(param["training_strategy"]):
        window = param["training_strategy"][window_index]
        if param["lr_option"]["reset"] == "y" and (j == 0):
            optimizer = optimizers[param['optim']](N.parameters(), weight_decay=param["weight_decay"], lr=param["lr_option"]["value"])

            lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param["max_lr"], max_steps=n_updates, type=param["lr_option"]["type"])

        [(TrainingInput, TrainingOutput, TrainingMasks, TrainingStd), (ValidationInput, ValidationOutput, ValidationMasks, ValidationStd)] = GetData(
            d_in=param['d_in'],
            n_pulse_plateau=param['n_pulse_plateau'],
            n_sat=param['n_sat'],
            n_mes=param['n_mes'],
            len_in=param['len_in'],
            len_out=param["len_out"],
            n_data_training=param['NDataT'],
            n_data_validation=param['NDataV'],
            sensitivity=param["sensitivity"],
            bias='freq',
            mean_min=window["mean"][0],
            mean_max=window["mean"][1],
            std_min=window["std"][0],
            std_max=window["std"][1],
            distrib=param["plot_distrib"],
            weight_f=weight_f,
            weight_l=weight_l,
            type='complete',
            save_path=data_dir,
            parallel=True
        )

        pbar = tqdm(total=n_iter_window, initial=j)
        t = time.time()
        while j < n_iter_window:

            error = 0
            time_to_observe = (int(j * param["FreqGradObs"]) == (j * param["FreqGradObs"])) and (param['FreqGradObs'] > 0)
            time_for_GIF = (j in torch.linspace(0, n_iter_window, abs(nb_frames_window), dtype=int)) and (nb_frames_GIF > 0)

            n_minibatch_epoch = n_minibatch - p
            while p < n_minibatch:
                InputMiniBatch = TrainingInput[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                OutputMiniBatch = TrainingOutput[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                TargetMaskMiniBatch = [TrainingMasks[0][p * mini_batch_size:(p + 1) * mini_batch_size].to(device),
                                       TrainingMasks[1][p * mini_batch_size:(p + 1) * mini_batch_size].to(device)]
                StdMiniBatch = TrainingStd[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                p += 1

                n_batch_epoch = n_batch - k
                while k < n_batch:
                    optimizer.zero_grad(set_to_none=True)

                    InputBatch = InputMiniBatch[k * batch_size:(k + 1) * batch_size]
                    OutputBatch = OutputMiniBatch[k * batch_size:(k + 1) * batch_size]
                    TargetMaskBatch = [TargetMaskMiniBatch[0][k * batch_size:(k + 1) * batch_size].to(device),
                                       TargetMaskMiniBatch[1][k * batch_size:(k + 1) * batch_size].to(device)]
                    StdBatch = StdMiniBatch[k * batch_size:(k + 1) * batch_size]
                    k += 1
                    Mask = TargetMaskBatch[1][:, :-1]

                    if param['error_weighting'] == 'n':
                        StdBatch = torch.mean(StdBatch)

                    Prediction = N(InputBatch, OutputBatch, TargetMaskBatch)[:, :-1, :]

                    err = torch.norm((Prediction - OutputBatch) / StdBatch, p=2) / sqrt((torch.sum(Mask) - batch_size) * d_out)
                    (param["mult_grad"] * err).backward()
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()

                    if k == 0 and p == 0 and time_to_observe:
                        DictGrad.update()

                    error += float(err) / (n_batch_epoch * n_minibatch_epoch)

                    if (time.time() - t > period_checkpoint) and (period_checkpoint > 0):
                        t = time.time()
                        try:
                            os.mkdir(save_path)
                        except:
                            pass
                        error = {"TrainingError": TrainingError,
                                 "ValidationError": ValidationError,
                                 "PlottingError": PlottingError}
                        saveObjAsXml(
                            {k: v for k, v in param.items() if not (k in ['resume_from'])},
                            os.path.join(save_path, "param"))
                        saveObjAsXml(error, os.path.join(save_path, "error"))
                        torch.save(best_state_dict, os.path.join(save_path, "Best_network"))
                        torch.save(N.state_dict().copy(), os.path.join(save_path, "Last_network"))
                        torch.save(weight_l, os.path.join(save_path, "WeightL"))
                        torch.save(weight_f, os.path.join(save_path, "WeightF"))
                        torch.save({
                            "scheduler_state_dict": lr_scheduler.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_hparams": lr_scheduler.get_hparams()
                        }, os.path.join(save_path, "Scheduler.pt"))

                        with open(os.path.join(save_path, "DictGrad.pkl"), "wb") as file:
                            pickle.dump(DictGrad, file)
                        with open(os.path.join(save_path, "ParamObs.pkl"), "wb") as file:
                            ParamObs = DictParamObserver(N)
                            pickle.dump(ParamObs, file)
                k = 0
            p = 0

            TrainingError.append(error)

            if time_to_observe:
                DictGrad.next(j)

            with torch.no_grad():
                Input = ValidationInput.to(device)
                Output = ValidationOutput.to(device)
                TargetMask = [ValidationMasks[0].to(device), ValidationMasks[1].to(device)]
                Std = ValidationStd.to(device)
                Mask = TargetMask[1][:, :-1]

                if param['error_weighting'] == 'n':
                    Std = torch.mean(Std)

                Prediction = N(Input, Output, TargetMask)[:, :-1, :]

                err = torch.norm((Prediction - Output) / Std, p=2) / sqrt((torch.sum(Mask) - NDataV) * d_out)
                ValidationError.append(float(err))

            if period_checkpoint == -1:
                if len(sys.argv) > 2:
                    if not 'progress_file' in locals():
                        progress_file = sys.argv[2]
                        # On crée / vide le fichier au début
                        with open(progress_file, "w") as f:
                            f.write("")
                    try:
                        with open(progress_file, "a") as f:
                            f.write(f"{j + n_iter_window * window_index} {ValidationError[-1] if ValidationError else float('inf')}\n")
                    except Exception as e:
                        print(f"[WARN] Could not write progress: {e}", flush=True)

            if error == min(TrainingError):
                best_state_dict = N.state_dict().copy()

            if time_for_GIF:
                with (torch.no_grad()):
                    Input = PlottingInput.to(device)
                    Output = PlottingOutput.to(device)
                    TargetMask = [PlottingMasks[0].to(device), PlottingMasks[1].to(device)]
                    Std = PlottingStd.to(device)
                    Mask = TargetMask[1][:, :-1]

                    if param['error_weighting'] == 'n':
                        Std = torch.mean(Std)

                    Prediction = N(Input, Output, TargetMask)[:, :-1, :]

                    err = torch.norm((Prediction - Output) / Std, p=2, dim=[-1, -2]) / torch.sqrt((torch.sum(Mask, dim=[1, 2]) - 1) * d_out)
                    PlottingError.append(err.reshape(res_GIF, res_GIF).tolist())


            j += 1
            pbar.n = j
            pbar.refresh()

        window_index += 1
        j = 0

    error = {"TrainingError": TrainingError,
             "ValidationError": ValidationError,
             "PlottingError": PlottingError}

    print(f"Final Error: {float(ValidationError[-1])}")

    if period_checkpoint != -1:
        saveObjAsXml(
            {k: v for k, v in param.items() if not (k in ['resume_from'])},
            os.path.join(save_path, "param"))
        saveObjAsXml(error, os.path.join(save_path, "error"))
        torch.save(best_state_dict, os.path.join(save_path, "Best_network"))
        torch.save(N.state_dict().copy(), os.path.join(save_path, "Last_network"))
        torch.save(weight_l, os.path.join(save_path, "WeightL"))
        torch.save(weight_f, os.path.join(save_path, "WeightF"))
        with open(os.path.join(save_path, "DictGrad.pkl"), "wb") as file:
            pickle.dump(DictGrad, file)
        with open(os.path.join(save_path, "ParamObs.pkl"), "wb") as file:
            ParamObs = DictParamObserver(N)
            pickle.dump(ParamObs, file)