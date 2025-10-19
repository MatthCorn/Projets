import optuna
import subprocess
import json
import os
import uuid
import time

import warnings
warnings.filterwarnings("ignore")

os.environ["MKL_THREADING_LAYER"] = "GNU"

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index("Projets"))], "Projets")
TRAINER_SCRIPT = {"global": os.path.join(local, "Inter", "NetworkGlobal", "Trainer.py"),
                  "windowed": os.path.join(local, "Inter", "NetworkGlobalWindowed", "Trainer.py"),
                  "memory": os.path.join(local, "Inter", "NetworkRecursive", "Trainer.py")}


def deep_suggest(trial, objet, key=None):
    if isinstance(objet, list):
        if objet[0] == 'suggest':
            trialtype, values, *kwargs = objet[1:]
            kwargs = dict(item for d in kwargs for item in d.items())

            if trialtype == 'categorical':
                return trial.suggest_categorical(key, values, **kwargs)
            if trialtype == 'int':
                return trial.suggest_int(key, *values, **kwargs)
            if trialtype == 'float':
                return trial.suggest_float(key, *values, **kwargs)
        else:
            return [deep_suggest(trial, x) for x in objet]
    if isinstance(objet, dict):
        return {key: deep_suggest(trial, value, key=key) for key, value in objet.items()}
    return objet


def objective(trial, RUN_DIR, params):
    params = deep_suggest(trial, params)

    # Nom unique pour le fichier JSON
    json_file = os.path.join(RUN_DIR, f"params_{uuid.uuid4().hex}.json")
    with open(json_file, "w") as f:
        json.dump(params, f)

    # Fichier temporaire pour la progression
    progress_file = os.path.join(RUN_DIR, f"progress_{uuid.uuid4().hex}.txt")

    # Lancement du script d'entraînement
    process = subprocess.Popen(
        ["python", TRAINER_SCRIPT[params['script']], json_file, progress_file],
        text=True,
        stdout=None,
        stderr=None,
        encoding="utf-8"
    )

    # Boucle de surveillance
    while process.poll() is None:
        time.sleep(1)  # toutes les 10 secondes
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                lines = f.readlines()
            if lines:
                try:
                    step, score = lines[-1].split()
                    step = int(step)
                    score = float(score)
                    trial.report(score, step)

                    if trial.should_prune():
                        process.terminate()
                        raise optuna.TrialPruned()
                except Exception:
                    pass

    score = float('inf')
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            lines = f.readlines()
        if lines:
            try:
                step, score = lines[-1].split()
                score = float(score)
            except:
                pass

    # Nettoyage
    stdout, _ = process.communicate()
    os.remove(json_file)
    if os.path.exists(progress_file):
        os.remove(progress_file)

    return score

if __name__ == "__main__":
    job_id = os.getenv("SLURM_JOB_ID", str(uuid.uuid4().hex))  # fallback si tu testes en local
    n_nodes = int(os.getenv("SLURM_NNODES", "1"))  # fallback si tu testes en local

    # Définir le répertoire de sauvegarde global
    run_dir = os.path.join(local, "Optuna", "Save", f"job_{job_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Définir le chemin de la base de données dans ce dossier
    db_path = os.path.join(run_dir, "optuna.db")
    storage = f"sqlite:///{db_path}"

    params = {
        "n_encoder": ['suggest', 'int', [2, 4]],
        "n_decoder": ['suggest', 'int', [2, 4]],
        "d_att": ['suggest', 'categorical', [64, 128]],
        "widths_embedding": ['suggest', 'categorical', [[16], [32]]],
        "width_FF": [128],
        "n_heads": ['suggest', 'categorical', [2, 4]],
        "dropout": ['suggest', 'float', [0, 0.3]],
        "lr_option": {
            "value": ['suggest', 'float', [1e-6, 1e-3], {'log': 1}],
            "reset": "y",
            "type": "cos"
        },
        "mult_grad": ["suggest", "int", [1e0, 1e4], {"log": 1}],
        "weight_decay": ['suggest', 'float', [1e-6, 1e-2], {'log': 1}],
        "batch_size": ['suggest', 'categorical', [512, 1024]],
        "n_iter": 10,  # pour optuna, on réduit un peu pour tester
        "NDataT": 5000,
        "NDataV": 100,
        "period_checkpoint": -1,
        "script": 'global',
        "n_trials": 3
    }

    try:
        import sys
        json_file = sys.argv[1]
        with open(json_file, "r") as f:
            temp_param = json.load(f)
        params.update(temp_param)
    except:
        print("nothing loaded")

    study = optuna.create_study(
        study_name="distributed-test",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner()
    )

    study.optimize(lambda x: objective(x, run_dir, params), n_trials=int(params['n_trials'] / n_nodes))

