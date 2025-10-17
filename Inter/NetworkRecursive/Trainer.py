from Inter.NetworkRecursive.DataMaker import GetData
from Inter.NetworkRecursive.Network import TransformerTranslator
from Complete.LRScheduler import Scheduler
from GradObserver.GradObserverClass import DictGradObserver
from Tools.ParamObs import DictParamObserver
import torch
from tqdm import tqdm
from math import sqrt
import time

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    ################################################################################################################################################
    # création des paramètres de la simulation

    param = {"n_encoder": 1,
             "n_decoder": 1,
             "len_in": 500,
             "len_out": 700,
             "len_in_window": 40,
             "len_out_window": 50,
             'size_tampon_source': 8,
             'size_tampon_target': 12,
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
                 "value": 3e-5,
                 "reset": "y",
                 "type": "cos"
             },
             "mult_grad": 10000,
             "weight_decay": 1e-3,
             "NDataT": 10000,
             "NDataV": 100,
             "batch_size": 1000,
             "n_iter": 70,
             "training_strategy": [
                 {"mean": [-5, 5], "std": [2, 5]},
             ],
             "distrib": "log",
             "plot_distrib": "log",
             "error_weighting": "y",
             "max_lr": 5,
             "FreqGradObs": 1 / 3,
             "warmup": 1,
             "resume_from": 'None',
             "max_inflight": 10,
             "period_checkpoint": 15 * 60,  # en secondes
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

    ################################################################################################################################################
    # pour les performances
    import psutil, sys, os

    p = psutil.Process(os.getpid())

    if sys.platform == "win32":
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    ################################################################################################################################################
    d_out = param['d_in'] + 1

    period_checkpoint = param["period_checkpoint"]
    nb_frames_GIF = param["nb_frames_GIF"]
    nb_frames_window = int(nb_frames_GIF / len(param["training_strategy"]))
    res_GIF = 10
    n_iter_window = int(param["n_iter"] / len(param["training_strategy"]))

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = TransformerTranslator(param['d_in'], d_out, d_att=param['d_att'], n_heads=param['n_heads'],
                              n_encoders=param['n_encoder'],
                              n_decoders=param['n_decoder'], widths_embedding=param['widths_embedding'],
                              len_in=param['len_in_window'], n_mes=param['n_mes'],
                              len_out=param['len_out_window'], norm=param['norm'],
                              dropout=param['dropout'], width_FF=param['width_FF'],
                              size_tampon_target=param['size_tampon_target'],
                              size_tampon_source=param['size_tampon_source'])

    N.to(device)

    NWindows = (param['len_in'] // (param['len_in_window'] - param['size_tampon_source']) + 1)
    NDataT = NWindows * param['NDataT']
    NDataV = NWindows * param['NDataV']
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
        PlottingInput, PlottingOutput, PlottingMemIn, PlottingMemOut, PlottingMasks, PlottingOutStd, PlottingMemStd = GetData(
            d_in=param['d_in'],
            n_pulse_plateau=param["n_pulse_plateau"],
            n_sat=param["n_sat"],
            n_mes=param["n_mes"],
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
            size_focus_source=param['len_in_window'] - param['size_tampon_source'],
            size_tampon_source=param['size_tampon_source'],
            size_tampon_target=param['size_tampon_target'],
            size_focus_target=param['len_out_window'] - param['size_tampon_target'],
            parallel=False
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
    save_dir = os.path.join(local, 'Inter', 'NetworkRecursive', 'Save')
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
        TrainingMemError = error_dict["TrainingMemError"]
        ValidationMemError = error_dict["ValidationMemError"]
        PlottingMemError = error_dict["PlottingMemError"]

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
        TrainingMemError = []
        ValidationMemError = []
        PlottingMemError = []

        window_index, j, p, k = 0, 0, 0, 0

    ################################################################################################################################################
    best_state_dict = N.state_dict().copy()
    t = time.time()
    while window_index < len(param["training_strategy"]):
        window = param["training_strategy"][window_index]
        if param["lr_option"]["reset"] == "y" and (j == 0):
            optimizer = optimizers[param['optim']](N.parameters(), weight_decay=param["weight_decay"], lr=param["lr_option"]["value"])

            lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param["max_lr"], max_steps=n_updates, type=param["lr_option"]["type"])

        [(TrainingInput, TrainingOutput, TrainingMemIn, TrainingMemOut, TrainingMasks, TrainingOutStd, TrainingMemStd),
         (ValidationInput, ValidationOutput, ValidationMemIn, ValidationMemOut, ValidationMasks, ValidationOutStd, ValidationMemStd)] = GetData(
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
            size_focus_source=param['len_in_window'] - param['size_tampon_source'],
            size_tampon_source=param['size_tampon_source'],
            size_tampon_target=param['size_tampon_target'],
            size_focus_target=param['len_out_window'] - param['size_tampon_target'],
            save_path=data_dir,
            parallel=True,
            max_inflight=param['max_inflight'],
        )

        pbar = tqdm(total=n_iter_window, initial=j)
        while j < n_iter_window:

            error = 0
            error_mem = 0
            time_to_observe = (int(j * param["FreqGradObs"]) == (j * param["FreqGradObs"])) and (param["FreqGradObs"] > 0)
            time_for_GIF = (j in torch.linspace(0, n_iter_window, abs(nb_frames_window), dtype=int)) and (nb_frames_GIF > 0)

            n_minibatch_epoch = n_minibatch - p
            while p < n_minibatch:
                InputMiniBatch = TrainingInput[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                OutputMiniBatch = TrainingOutput[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                MemInMiniBatch = TrainingMemIn[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                MemOutMiniBatch = TrainingMemOut[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                MaskMiniBatch = [mask[p * mini_batch_size:(p + 1) * mini_batch_size].to(device) for mask in
                                 TrainingMasks]
                OutStdMiniBatch = TrainingOutStd[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                MemStdMiniBatch = TrainingMemStd[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                p += 1

                n_batch_epoch = n_batch - k
                while k < n_batch:
                    optimizer.zero_grad(set_to_none=True)

                    InputBatch = InputMiniBatch[k * batch_size:(k + 1) * batch_size]
                    OutputBatch = OutputMiniBatch[k * batch_size:(k + 1) * batch_size]
                    MemInBatch = MemInMiniBatch[k * batch_size:(k + 1) * batch_size]
                    MemOutBatch = MemOutMiniBatch[k * batch_size:(k + 1) * batch_size]
                    MaskBatch = [mask[k * batch_size:(k + 1) * batch_size].to(device) for mask in MaskMiniBatch]
                    OutStdBatch = OutStdMiniBatch[k * batch_size:(k + 1) * batch_size]
                    MemStdBatch = MemStdMiniBatch[k * batch_size:(k + 1) * batch_size]
                    k += 1

                    if param['error_weighting'] == 'n':
                        StdBatch = torch.mean(StdBatch)

                    InputMask = MaskBatch[:-1]
                    WindowMask = MaskBatch[-1]

                    Prediction, PredictionMemOut = N(InputBatch, OutputBatch, MemInBatch, InputMask)
                    Prediction = Prediction[:, :-1, :]

                    err = torch.norm((Prediction - OutputBatch) / OutStdBatch * WindowMask, p=2) / (
                                abs(WindowMask.sum() - batch_size) * d_out).sqrt()

                    err_mem = torch.norm((PredictionMemOut - MemOutBatch) / MemStdBatch) / sqrt(
                            MemOutBatch.shape[0] * MemOutBatch.shape[1] * MemOutBatch.shape[2])

                    (param["mult_grad"] * (err + err_mem)).backward()
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()

                    if k == 0 and p == 0 and time_to_observe:
                        DictGrad.update()

                    error += float(err) / (n_batch_epoch * n_minibatch_epoch)
                    error_mem += float(err_mem) / (n_batch_epoch * n_minibatch_epoch)

                    if (time.time() - t > period_checkpoint) and (period_checkpoint > 0):
                        t = time.time()
                        try:
                            os.mkdir(save_path)
                        except:
                            pass
                        error_dict = {"TrainingError": TrainingError,
                                      "ValidationError": ValidationError,
                                      "PlottingError": PlottingError,
                                      "TrainingMemError": TrainingMemError,
                                      "ValidationMemError": ValidationMemError,
                                      "PlottingMemError": PlottingMemError
                                      }
                        saveObjAsXml(
                            {k: v for k, v in param.items() if k != 'resume_from'},
                            os.path.join(save_path, "param"))
                        saveObjAsXml(error_dict, os.path.join(save_path, "error"))
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
            
            if time_to_observe:
                DictGrad.next(j)

            with torch.no_grad():
                Input = ValidationInput.to(device)
                Output = ValidationOutput.to(device)
                MemIn = ValidationMemIn.to(device)
                MemOut = ValidationMemOut.to(device)

                Mask = [mask.to(device) for mask in ValidationMasks]
                OutStd = ValidationOutStd.to(device)
                MemStd = ValidationMemStd.to(device)

                if param['error_weighting'] == 'n':
                    Std = torch.mean(Std)

                InputMask = Mask[:-1]
                WindowMask = Mask[-1]
                Prediction, PredictionMemOut = N(Input, Output, MemIn, InputMask)
                Prediction = Prediction[:, :-1, :]

                err = torch.norm((Prediction - Output) / OutStd * WindowMask, p=2) / (
                            (WindowMask.sum() - NDataV) * d_out).sqrt()

                err_mem = torch.norm((PredictionMemOut - MemOut) / MemStd) / sqrt(
                    MemOut.shape[0] * MemOut.shape[1] * MemOut.shape[2])

                ValidationError.append(float(err))
                ValidationMemError.append(float(err_mem))

            TrainingError.append(error)
            TrainingMemError.append(error_mem)

            if error == min(TrainingError):
                best_state_dict = N.state_dict().copy()

            if time_for_GIF:
                with torch.no_grad():
                    Input = PlottingInput.to(device)
                    Output = PlottingOutput.to(device)
                    MemIn = PlottingMemIn.to(device)
                    MemOut = PlottingMemOut.to(device)
                    Mask = [mask.to(device) for mask in PlottingMasks]
                    OutStd = PlottingOutStd.to(device)
                    MemStd = PlottingMemStd.to(device)

                    if param['error_weighting'] == 'n':
                        Std = torch.mean(Std)

                    InputMask = Mask[:-1]
                    WindowMask = Mask[-1]
                    Prediction, PredictionMemOut = N(Input, Output, MemIn, InputMask)
                    Prediction = Prediction[:, :-1, :]

                    err = torch.norm((Prediction - Output) * WindowMask / OutStd, p=2, dim=[-1, -2]) / (
                            (WindowMask.sum(dim=[-1, -2]) - 1) * d_out).sqrt()
                    err = err.reshape(-1, NWindows).mean(dim=-1)

                    err_mem = torch.norm((PredictionMemOut - MemOut) / OutStd, p=2, dim=[-1, -2]) / sqrt(
                            MemOut.shape[1] * MemOut.shape[1])
                    err_mem = err_mem.reshape(-1, NWindows).mean(dim=-1)

                    PlottingError.append(err.reshape(res_GIF, res_GIF).tolist())
                    PlottingMemError.append(err_mem.reshape(res_GIF, res_GIF).tolist())

            j += 1
            pbar.n = j
            pbar.refresh()

        window_index += 1

    error_dict = {"TrainingError": TrainingError,
                  "ValidationError": ValidationError,
                  "PlottingError": PlottingError,
                  "TrainingMemError": TrainingMemError,
                  "ValidationMemError": ValidationMemError,
                  "PlottingMemError": PlottingMemError
                  }

    if period_checkpoint != -1:
        saveObjAsXml(
            {k: v for k, v in param.items() if k != 'resume_from'},
            os.path.join(save_path, "param"))
        saveObjAsXml(error_dict, os.path.join(save_path, "error"))
        torch.save(best_state_dict, os.path.join(save_path, "Best_network"))
        torch.save(N.state_dict().copy(), os.path.join(save_path, "Last_network"))
        torch.save(weight_l, os.path.join(save_path, "WeightL"))
        torch.save(weight_f, os.path.join(save_path, "WeightF"))
        with open(os.path.join(save_path, "DictGrad.pkl"), "wb") as file:
            pickle.dump(DictGrad, file)
        with open(os.path.join(save_path, "ParamObs.pkl"), "wb") as file:
            ParamObs = DictParamObserver(N)
            pickle.dump(ParamObs, file)