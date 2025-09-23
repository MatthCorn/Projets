from Inter.NetworkGlobalWindowed.SpecialUtils import GetData
from Inter.NetworkGlobalWindowed.Network import TransformerTranslator
from Complete.LRScheduler import Scheduler
from GradObserver.GradObserverClass import DictGradObserver
from Tools.ParamObs import DictParamObserver
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, inputs, outputs, masks, stds):
        self.inputs = inputs
        self.outputs = outputs
        self.masks = masks
        self.stds = stds

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return (self.inputs[idx],
                self.outputs[idx],
                [mask[idx] for mask in self.masks],
                self.stds[idx])

import platform


def auto_num_workers(max_cap=None):
    """
    Retourne un nombre de workers adapté au système et au CPU.

    Args:
        max_cap (int, optional): limite supérieure (par défaut dépend de l'OS).

    Returns:
        int: valeur recommandée pour num_workers
    """
    n_cpu = os.cpu_count() or 1
    print(n_cpu)
    system = platform.system()

    if system == "Windows":
        # Windows → spawn est lourd, donc on limite
        default_cap = 8
    else:
        # Linux/macOS → fork est plus léger
        default_cap = n_cpu

    if max_cap is None:
        max_cap = default_cap

    # règle empirique : on garde 1 cœur pour le système
    n_workers = max(0, min(n_cpu - 1, max_cap))
    return n_workers


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    ################################################################################################################################################
    # création des paramètres de la simulation

    param = {"n_encoder": 10,
             "n_decoder": 10,
             "len_in": 500,
             "len_out": 700,
             "len_in_window": 20,
             "len_out_window": 30,
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
             "n_iter": 100,
             "training_strategy": [
                 {"mean": [-5, 5], "std": [0.2, 1]},
             ],
             "distrib": "log",
             "plot_distrib": "log",
             "error_weighting": "y",
             "max_lr": 5,
             "FreqGradObs": 1 / 3,
             "warmup": 1,
             "resume_from": 'r'}

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

    freq_checkpoint = 1/10
    nb_frames_GIF = 100
    nb_frames_window = int(nb_frames_GIF / len(param["training_strategy"]))
    res_GIF = 10
    n_iter_window = int(param["n_iter"] / len(param["training_strategy"]))

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = TransformerTranslator(param['d_in'], d_out, d_att=param['d_att'], n_heads=param['n_heads'],
                              n_encoders=param['n_encoder'],
                              n_decoders=param['n_decoder'], widths_embedding=param['widths_embedding'],
                              len_in=param['len_in_window'],
                              len_out=param['len_out_window'], norm=param['norm'], dropout=param['dropout'],
                              width_FF=param['width_FF'],
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
    save_dir = os.path.join(local, 'Inter', 'NetworkGlobalWindowed', 'Save')
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

        j = len(TrainingError) % n_iter_window
        window_index = len(TrainingError) // n_iter_window
        print(f"Reprise à la fenêtre {window_index}, itération {j}")

    except Exception as e:
        print(f"Erreur lors de la reprise du checkpoint : {e}")
        print("Lancement d'un entraînement depuis zéro.")


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

        window_index = 0
        j = 0

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
            size_focus_source=param['len_in_window'] - param['size_tampon_source'],
            size_tampon_source=param['size_tampon_source'],
            size_tampon_target=param['size_tampon_target'],
            size_focus_target=param['len_out_window'] - param['size_tampon_target'],
            save_path=data_dir,
            parallel=True,
            max_inflight=500,
        )

        train_dataset = CustomDataset(TrainingInput, TrainingOutput, TrainingMasks, TrainingStd)
        val_dataset = CustomDataset(ValidationInput, ValidationOutput, ValidationMasks, ValidationStd)

        train_loader = DataLoader(
            train_dataset,
            batch_size=param["batch_size"],
            shuffle=True,
            num_workers=auto_num_workers(),
            pin_memory=True if device.type == "cuda" else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=param["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True if device.type == "cuda" else False
        )

        pbar = tqdm(total=n_iter_window, initial=j)
        while j < n_iter_window:

            error = 0
            time_to_observe = (int(j * param["FreqGradObs"]) == (j * param["FreqGradObs"]))
            time_for_checkpoint = (int(j * freq_checkpoint) == (j * freq_checkpoint))

            for batch in train_loader:
                InputBatch, OutputBatch, MaskBatch, StdBatch = batch

                InputBatch = InputBatch.to(device)
                OutputBatch = OutputBatch.to(device)
                MaskBatch = [mask.to(device) for mask in MaskBatch]
                StdBatch = StdBatch.to(device)

                if param['error_weighting'] == 'n':
                    StdBatch = torch.mean(StdBatch)

                InputMask = MaskBatch[:-1]
                WindowMask = MaskBatch[-1]

                optimizer.zero_grad(set_to_none=True)
                Prediction = N(InputBatch, OutputBatch, InputMask)[:, :-1, :]

                err = torch.norm((Prediction - OutputBatch) / StdBatch * WindowMask, p=2) / (
                        abs(WindowMask.sum() - InputBatch.shape[0]) * d_out).sqrt()

                (param["mult_grad"] * err).backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                if time_to_observe:
                    DictGrad.update()

                error += float(err) / len(train_loader)

            if time_to_observe:
                DictGrad.next(j)

            with torch.no_grad():
                val_error = 0
                for batch in val_loader:
                    InputBatch, OutputBatch, MaskBatch, StdBatch = batch

                    InputBatch = InputBatch.to(device)
                    OutputBatch = OutputBatch.to(device)
                    MaskBatch = [mask.to(device) for mask in MaskBatch]
                    StdBatch = StdBatch.to(device)

                    if param['error_weighting'] == 'n':
                        StdBatch = torch.mean(StdBatch)

                    InputMask = MaskBatch[:-1]
                    WindowMask = MaskBatch[-1]

                    Prediction = N(InputBatch, OutputBatch, InputMask)[:, :-1, :]
                    err = torch.norm((Prediction - OutputBatch) / StdBatch * WindowMask, p=2) / (
                            (WindowMask.sum() - InputBatch.shape[0]) * d_out).sqrt()
                    val_error += float(err)

                ValidationError.append(val_error / len(val_loader))

            TrainingError.append(error)

            if error == min(TrainingError):
                best_state_dict = N.state_dict().copy()

            if time_for_checkpoint:
                try:
                    os.mkdir(save_path)
                except:
                    pass
                error_dict = {"TrainingError": TrainingError,
                         "ValidationError": ValidationError,
                         "PlottingError": PlottingError}
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

            j += 1
            pbar.n = j
            pbar.refresh()

        window_index += 1

    error_dict = {"TrainingError": TrainingError,
             "ValidationError": ValidationError,
             "PlottingError": PlottingError}
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