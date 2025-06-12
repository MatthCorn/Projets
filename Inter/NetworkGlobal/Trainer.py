from Inter.Model.DataMaker import GetData
from Inter.NetworkGlobal.Network import TransformerTranslator
from Complete.LRScheduler import Scheduler
from GradObserver.GradObserverClass import DictGradObserver
from Tools.ParamObs import DictParamObserver
from math import sqrt
import torch
from tqdm import tqdm

if __name__ == '__main__':
    ################################################################################################################################################
    # pour sauvegarder toutes les informations de l'apprentissage
    import os
    import datetime
    from Tools.XMLTools import saveObjAsXml
    import pickle
    import time

    local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index("Projets"))], "Projets")
    base_folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
    save_dir = os.path.join(local, 'Inter', 'NetworkGlobal', 'Save')
    data_dir = os.path.join(local, 'Inter', 'Data')

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

    param = {"n_encoder": 10,
             "n_decoder": 10,
             "len_in": 10,
             "len_out": 20,
             "n_pulse_plateau": 6,
             "n_sat": 5,
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
             "NDataT": 1000000,
             "NDataV": 1000,
             "batch_size": 1000,
             "n_iter": 100,
             "training_strategy": [
                 {"mean": [-5, 5], "std": [0.2, 1]},
             ],
             "distrib": "log",
             "plot_distrib": "log",
             "error_weighting": "y",
             "max_lr": 5,
             "FreqGradObs": 1/3,
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
    d_out = param['d_in'] + 1

    freq_checkpoint = 1/10
    nb_frames_GIF = 100
    nb_frames_window = int(nb_frames_GIF / len(param["training_strategy"]))
    res_GIF = 50
    n_iter_window = int(param["n_iter"] / len(param["training_strategy"]))

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = TransformerTranslator(param['d_in'], d_out, d_att=param['d_att'], n_heads=param['n_heads'], n_encoders=param['n_encoder'],
                              n_decoders=param['n_decoder'], widths_embedding=param['widths_embedding'], len_in=param['len_in'],
                              len_out=param['len_out'], norm=param['norm'], dropout=param['dropout'], width_FF=param['width_FF'])
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
    TrainingError = []
    ValidationError = []
    PlottingError = []

    n_updates = int(NDataT / batch_size) * n_iter
    warmup_steps = int(NDataT / batch_size * param["warmup"])
    lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param["max_lr"], max_steps=n_updates, type=param["lr_option"]["type"])

    PlottingInput, PlottingOutput, PlottingMasks, PlottingStd = GetData(
        d_in=param['d_in'],
        n_pulse_plateau=param["n_pulse_plateau"],
        n_sat=param["n_sat"],
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

    best_state_dict = N.state_dict().copy()

    for window in param["training_strategy"]:
        if param["lr_option"]["reset"] == "y":
            optimizer = optimizers[param['optim']](N.parameters(), weight_decay=param["weight_decay"], lr=param["lr_option"]["value"])

            n_updates = int(NDataT / batch_size) * n_iter_window
            warmup_steps = int(NDataT / batch_size * param["warmup"])
            lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param["max_lr"], max_steps=n_updates, type=param["lr_option"]["type"])

        [(TrainingInput, TrainingOutput, TrainingMasks, TrainingStd), (ValidationInput, ValidationOutput, ValidationMasks, ValidationStd)] = GetData(
            d_in=param['d_in'],
            n_pulse_plateau=param['n_pulse_plateau'],
            n_sat=param['n_sat'],
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

        for j in tqdm(range(n_iter_window)):
            error = 0
            time_to_observe = (int(j * param["FreqGradObs"]) == (j * param["FreqGradObs"]))
            time_for_checkpoint = (int(j * freq_checkpoint) == (j * freq_checkpoint))
            time_for_GIF = (j in torch.linspace(0, n_iter_window, nb_frames_window, dtype=int))

            for p in range(n_minibatch):
                InputMiniBatch = TrainingInput[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                OutputMiniBatch = TrainingOutput[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                TargetMaskMiniBatch = [TrainingMasks[0][p * mini_batch_size:(p + 1) * mini_batch_size].to(device),
                                       TrainingMasks[1][p * mini_batch_size:(p + 1) * mini_batch_size].to(device)]
                StdMiniBatch = TrainingStd[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)

                for k in range(n_batch):
                    optimizer.zero_grad(set_to_none=True)

                    InputBatch = InputMiniBatch[k * batch_size:(k + 1) * batch_size]
                    OutputBatch = OutputMiniBatch[k * batch_size:(k + 1) * batch_size]
                    TargetMaskBatch = [TargetMaskMiniBatch[0][k * batch_size:(k + 1) * batch_size].to(device),
                                       TargetMaskMiniBatch[1][k * batch_size:(k + 1) * batch_size].to(device)]
                    StdBatch = StdMiniBatch[k * batch_size:(k + 1) * batch_size]

                    if param['error_weighting'] == 'n':
                        StdBatch = torch.mean(StdBatch)

                    Prediction = N(InputBatch, OutputBatch, TargetMaskBatch)[:, :-1, :]

                    err = torch.norm((Prediction - OutputBatch) / StdBatch, p=2) / sqrt(batch_size * d_out * NOutput)
                    (param["mult_grad"] * err).backward()
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()

                    if k == 0 and time_to_observe:
                        DictGrad.update()

                    error += float(err) / (n_batch * n_minibatch)

            if time_to_observe:
                DictGrad.next(j)

            with torch.no_grad():
                Input = ValidationInput.to(device)
                Output = ValidationOutput.to(device)
                TargetMask = [ValidationMasks[0].to(device), ValidationMasks[1].to(device)]
                Std = ValidationStd.to(device)

                if param['error_weighting'] == 'n':
                    Std = torch.mean(Std)

                Prediction = N(Input, Output, TargetMask)[:, :-1, :]

                err = torch.norm((Prediction - Output) / Std, p=2) / sqrt(NDataV * d_out * NOutput)
                ValidationError.append(float(err))

            TrainingError.append(error)

            if error == min(TrainingError):
                best_state_dict = N.state_dict().copy()

            if time_for_GIF:
                with torch.no_grad():
                    Input = PlottingInput.to(device)
                    Output = PlottingOutput.to(device)
                    TargetMask = [PlottingMasks[0].to(device), PlottingMasks[1].to(device)]
                    Std = PlottingStd.to(device)

                    if param['error_weighting'] == 'n':
                        Std = torch.mean(Std)

                    Prediction = N(Input, Output, TargetMask)[:, :-1, :]

                    err = torch.norm((Prediction - Output) / Std, p=2, dim=[-1, -2]) / sqrt(d_out * NOutput)
                    PlottingError.append(err.reshape(res_GIF, res_GIF).tolist())

            if time_for_checkpoint:
                try:
                    os.mkdir(save_path)
                except:
                    pass
                error = {"TrainingError": TrainingError,
                         "ValidationError": ValidationError,
                         "PlottingError": PlottingError}
                saveObjAsXml(param, os.path.join(save_path, "param"))
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

    error = {"TrainingError": TrainingError,
             "ValidationError": ValidationError,
             "PlottingError": PlottingError}
    saveObjAsXml(param, os.path.join(save_path, "param"))
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