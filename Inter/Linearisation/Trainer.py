from Inter.Linearisation.GRU import GRUNetwork
from Inter.Linearisation.SpecialUtils import GetData
from Inter.Linearisation.TCNetwork import TCNet
from Inter.Linearisation.LSTMNetwork import LSTMWithAttention as LSTM
from Inter.Linearisation.TrfNetwork import Transformer
from Complete.LRScheduler import Scheduler
from math import sqrt
import torch
from tqdm import tqdm
import time
from Tools.MCCutils import soft_mcc
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)  # permet les lancements en parallèle pour l'optimisation de l'architecture avec Optuna

    ################################################################################################################################################
    ###                                              création des paramètres de la simulation                                                    ###
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

    # On met à jour les paramètres avec ceux qui peuvent être passés en argument lors de l'exécution du script
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
    ###     permet d'afficher les informations du composant sur lequel le script s'exécute, pratique pour débug les lancements en parallèle      ###
    import os
    try:
        gpu_id = torch.cuda.current_device()
        print(f"[Worker Node {os.getenv('SLURM_NODEID')} | GPU {gpu_id}] Starting training", flush=True)
    except:
        pass
    ################################################################################################################################################
    ###  permet sur certains ordinateurs (mon ordi portable) d'avoir plus de performance en forçant une priorité haute à l'exécution du script   ###
    import psutil, sys, os

    p = psutil.Process(os.getpid())

    if sys.platform == "win32":
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    ################################################################################################################################################
    ###                                    on initialise les paramètres et le réseau de neurones                                                 ###

    Network = {'TCN': TCNet, 'LSTM': LSTM, 'Transformer': Transformer, 'GRU': GRUNetwork}[param['network']]

    d_out = param['d_in']

    period_checkpoint = param["period_checkpoint"]  # 0 : pas de checkpoint en entrainement, -1 : pas de sauvegarde du tout
    n_iter_window = int(param["n_iter"] / len(param["training_strategy"])) # nombre d'itération par fenêtre d'entrainement (curriculum learning)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = Network(
        param['d_in'],
        param['d_in'],
        param['d_att'],
        param['d_in'],
        n_layers=param['n_layers'],
        n_head=param['n_heads'],
        mem_length=param['mem_length'],
        max_len=param['len_in'] + param['len_out'],
        kernel_size=param['kernel_size'],
        dropout=param['dropout'],
    )

    N.to(device)
    print(sum(p.numel() for p in N.parameters()))

    NDataT = param['NDataT']
    NDataV = param['NDataV']
    DInput = param['d_in']
    NInput = param['len_in']
    NOutput = param['len_out']
    weight_f = torch.tensor([1., 0.] + [0.] * (param['d_in'] - 4)).numpy()
    weight_l = torch.tensor([0., 1.] + [0.] * (param['d_in'] - 4)).numpy()

    mini_batch_size = 50000
    n_minibatch = int(NDataT / mini_batch_size)
    batch_size = param["batch_size"]
    n_batch = int(mini_batch_size / batch_size)

    n_iter = param["n_iter"]

    # on peut choisir de réinitialiser le taux d'apprentissage à chaque fenêtre,
    # ce qui modifie les paramètres qui règlent le lr_scheduler
    if param['lr_option']['reset'] == 'y':
        n_updates = int(NDataT / batch_size) * n_iter_window
        warmup_steps = int(NDataT / batch_size * param["warmup"])
    else:
        n_updates = int(NDataT / batch_size) * n_iter
        warmup_steps = int(NDataT / batch_size * param["warmup"])

    optimizers = {
        "AdamW": torch.optim.AdamW,
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }

    ################################################################################################################################################
    ###         on va ici initialiser l'état de l'entrainement, soit à partir d'un entrainement qu'on continue, soit à partir de rien            ###

    from Tools.XMLTools import saveObjAsXml
    local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index("Projets"))], "Projets")
    save_dir = os.path.join(local, 'Inter', 'Linearisation', 'Save')
    data_dir = os.path.join(local, 'Inter', 'Data')

    # on essaie de charger un ancien entrainement. pour que ça marche, le chemin spécifié dans "resume_from" doit exister
    # et les paramètres de l'architecture doivent être compatible avec ceux de l'ancien entrainement
    try:
        from Tools.XMLTools import loadXmlAsObj
        resume_from = param["resume_from"]

        save_path = os.path.join(save_dir, resume_from)

        print(f"Reprise à partir du checkpoint : {save_path}")
        N.load_state_dict(torch.load(os.path.join(save_path, "Last_network")))
        best_state_dict = torch.load(os.path.join(save_path, "Best_network"))

        checkpoint = torch.load(os.path.join(save_path, "Scheduler.pt"))

        optimizer = optimizers[param['optim']](N.parameters(), weight_decay=param["weight_decay"], lr=param["lr_option"]["value"])

        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

        lr_scheduler = Scheduler(optimizer=optimizer, **checkpoint["scheduler_hparams"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        error_dict = loadXmlAsObj(os.path.join(save_path, "error"))
        TrainingError = error_dict["TrainingError"]
        TrainingErrorNext = error_dict["TrainingErrorNext"]
        ValidationError = error_dict["ValidationError"]
        ValidationErrorNext = error_dict["ValidationErrorNext"]

        # à partir de l'état du scheduler, on peut retrouver où on était dans l'entrainement
        # l'état d'avancement de l'entrainement est défini par window_index, j, p, k
        window_index, r = divmod(checkpoint['scheduler_state_dict']['last_epoch'] + 1, n_iter_window * n_batch * n_minibatch)
        j, r = divmod(r, n_batch * n_minibatch)
        p, k = divmod(r, n_batch)

        print(f"Reprise à la fenêtre {window_index}, itération {j}")

    except Exception as e:
        print(f"Erreur lors de la reprise du checkpoint : {e}")
        print("Lancement d'un entraînement depuis zéro.")

        # si on n'a pas d'entrainement à reprendre, on va créer un dossier de sauvegarde
        if period_checkpoint != -1:
            # pour sauvegarder toutes les informations de l'apprentissage
            import datetime
            import time

            base_folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")

            # on crée une boucle pour trouver un nom de dossier libre, c'est nécessaire pour gérer la concurrence
            # d'entrainements lancés simultanément, qui aurait le même base_folder
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

        optimizer = optimizers[param['optim']](N.parameters(), weight_decay=param["weight_decay"], lr=param["lr_option"]["value"])
        lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param["max_lr"], max_steps=n_updates, type=param["lr_option"]["type"])

        TrainingError = []
        TrainingErrorNext = []
        ValidationError = []
        ValidationErrorNext = []

        window_index, j, p, k = 0, 0, 0, 0

        best_state_dict = N.state_dict().copy()

    ################################################################################################################################################
    ###                                         on commence la procédure d'entrainement ici                                                      ###

    # les boucles de l'entrainement sont des while avec des indices incrémentés pour pouvoir reprendre l'entrainement en cours facilement
    while window_index < len(param["training_strategy"]):
        window = param["training_strategy"][window_index]

        # si on commence l'entrainement sur la fenêtre et que le paramétrage demande un reset des params de l'optimisation, on les réinitialise
        if param["lr_option"]["reset"] == "y" and (j == 0):
            optimizer = optimizers[param['optim']](N.parameters(), weight_decay=param["weight_decay"], lr=param["lr_option"]["value"])
            lr_scheduler = Scheduler(optimizer, 256, warmup_steps, max=param["max_lr"], max_steps=n_updates, type=param["lr_option"]["type"])

        # on charge les données d'entrainement et de validation de la fenêtre
        [(TrainingInput1, TrainingInput2, TrainingOutput, TrainingStd,
          TrainingNextMaskInput, TrainingNextMaskOutput, TrainingOnSequenceMask),
         (ValidationInput1, ValidationInput2, ValidationOutput, ValidationStd,
          ValidationNextMaskInput, ValidationNextMaskOutput, ValidationOnSequenceMask)] = GetData(
            d_in=param['d_in']-1,
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
            save_path=data_dir,
            parallel=True
        )

        pbar = tqdm(total=n_iter_window, initial=j)
        t = time.time()  # sert à enregistrer régulièrement des checkpoints de l'entrainement
        while j < n_iter_window:  # traitement de la fenêtre

            error = 0
            error_next = 0

            n_minibatch_epoch = n_minibatch - p

            # on découpe chaque fenêtre une première fois (mini_batch) pour faciliter le chargement des données sur le GPU
            while p < n_minibatch:
                Input1MiniBatch = TrainingInput1[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                Input2MiniBatch = TrainingInput2[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                OutputMiniBatch = TrainingOutput[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                NMInputMiniBatch = TrainingNextMaskInput[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                NMOutputMiniBatch = TrainingNextMaskOutput[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                OSMMiniBatch = TrainingOnSequenceMask[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)

                StdMiniBatch = TrainingStd[p * mini_batch_size:(p + 1) * mini_batch_size].to(device)
                p += 1

                n_batch_epoch = n_batch - k

                # on découpe chaque minibatch en batch pour faire la mise-à-jour des paramètres
                while k < n_batch:
                    optimizer.zero_grad(set_to_none=True)

                    Input1Batch = Input1MiniBatch[k * batch_size:(k + 1) * batch_size]
                    Input2Batch = Input2MiniBatch[k * batch_size:(k + 1) * batch_size]
                    OutputBatch = OutputMiniBatch[k * batch_size:(k + 1) * batch_size]
                    NMInputBatch = NMInputMiniBatch[k * batch_size:(k + 1) * batch_size]
                    NMOutputBatch = NMOutputMiniBatch[k * batch_size:(k + 1) * batch_size]
                    OSMBatch = OSMMiniBatch[k * batch_size:(k + 1) * batch_size]

                    StdBatch = StdMiniBatch[k * batch_size:(k + 1) * batch_size]
                    k += 1

                    if param['error_weighting'] == 'n':
                        StdBatch = torch.mean(StdBatch)

                    # calcul de l'erreur de la prédiction et update des poids du réseau de neurones
                    Prediction, is_next = N(Input1Batch, Input2Batch, NMInputBatch)

                    err = (torch.norm((Prediction - OutputBatch) * (1 - NMOutputBatch) * OSMBatch / StdBatch, p=2) /
                           sqrt((torch.sum((1 - NMOutputBatch) * OSMBatch) - batch_size) * d_out))
                    err_next = torch.mean(1 - soft_mcc(is_next, NMOutputBatch, OSMBatch))

                    (param["mult_grad"] * ((1 - param['weight_error']) * err + param['weight_error'] * err_next)).backward()
                    torch.nn.utils.clip_grad_norm_(N.parameters(), 1.0)
                    optimizer.step()

                    if lr_scheduler is not None:  # update du scheduler
                        lr_scheduler.step()

                    error += float(err) / (n_batch_epoch * n_minibatch_epoch)
                    error_next += float(err_next) / (n_batch_epoch * n_minibatch_epoch)

                    if (time.time() - t > period_checkpoint) and (period_checkpoint > 0):  # enregistrement du checkpoint
                        t = time.time()
                        try:
                            os.mkdir(save_path)
                        except:
                            pass
                        error_dict = {"TrainingError": TrainingError,
                                      "ValidationError": ValidationError}
                        saveObjAsXml({k: v for k, v in param.items() if not (k in ['resume_from'])},os.path.join(save_path, "param"))
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

                k = 0
            p = 0

            TrainingError.append(error)
            TrainingErrorNext.append(error_next)

            # calcul de l'erreur de validation
            with torch.no_grad():
                Input1 = ValidationInput1.to(device)
                Input2 = ValidationInput2.to(device)
                Output = ValidationOutput.to(device)
                NMInput = ValidationNextMaskInput.to(device)
                NMOutput = ValidationNextMaskOutput.to(device)
                OSM = ValidationOnSequenceMask.to(device)

                Std = ValidationStd.to(device)
                k += 1

                if param['error_weighting'] == 'n':
                    Std = torch.mean(Std)

                N.eval()
                Prediction, is_next = N(Input1, Input2, NMInput)
                N.train()

                err = torch.norm((Prediction - Output) * (1 - NMOutput) * OSM / Std, p=2) / sqrt((torch.sum((1 - NMOutput) * OSM) - NDataV) * d_out)
                err_next = torch.mean((1 - soft_mcc(is_next, NMOutput, OSM)) / Std) * torch.mean(Std)

                ValidationError.append(float(err))
                ValidationErrorNext.append(float(err_next))

            # écriture de l'erreur de validation dans un fichier à part, uniquement utile pour qu'Optuna
            # puisse suivre l'entrainement et le couper prématurément si besoin
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

            if error == min(TrainingError):  # mise-à-jour des meilleurs paramètres
                best_state_dict = N.state_dict().copy()

            j += 1
            pbar.n = j
            pbar.refresh()

        window_index += 1
        j = 0

    ################################################################################################################################################
    ###                                         enregistrement finale des infos de l'entrainement                                                ###

    error_dict = {"TrainingError": TrainingError,
                  "TrainingErrorNext": TrainingErrorNext,
                  "ValidationError": ValidationError,
                  "ValidationErrorNext": ValidationErrorNext}

    print(f"Final Error: {float(ValidationError[-1])}")

    if period_checkpoint != -1:
        saveObjAsXml({k: v for k, v in param.items() if not (k in ['resume_from'])},os.path.join(save_path, "param"))
        saveObjAsXml(error_dict, os.path.join(save_path, "error"))
        torch.save(best_state_dict, os.path.join(save_path, "Best_network"))
        torch.save(N.state_dict().copy(), os.path.join(save_path, "Last_network"))
        torch.save(weight_l, os.path.join(save_path, "WeightL"))
        torch.save(weight_f, os.path.join(save_path, "WeightF"))
