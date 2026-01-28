import optuna
import subprocess
import json
import os
import uuid
import time
import ast

import warnings
warnings.filterwarnings("ignore")

os.environ["MKL_THREADING_LAYER"] = "GNU"

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index("Projets"))], "Projets")

def deep_suggest(trial, objet, key=None):
    if isinstance(objet, list):
        if objet[0] == 'suggest':
            trialtype, values, *kwargs = objet[1:]
            kwargs = dict(item for d in kwargs for item in d.items())

            if trialtype == 'categorical':
                if isinstance(values[0], list):
                    return ast.literal_eval(trial.suggest_categorical(key, [str(x) for x in values], **kwargs))
                else:
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
        ["python", os.path.join(local, *params['script']), json_file, progress_file],
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
                step, score = lines[-1].split()
                step = int(step)
                score = float(score)
                trial.report(score, step)

                try:
                    if trial.should_prune() and params['prune']:
                        process.terminate()
                        process.wait(timeout=30)
                except Exception as e:
                    print(e)
                    try:
                        process.kill()
                        process.wait()
                    except Exception as e:
                        print(e)

    if trial.should_prune() and params['prune']:
        # Nettoyage
        os.remove(json_file)
        if os.path.exists(progress_file):
            os.remove(progress_file)
        raise optuna.TrialPruned()

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
    params = {
        "lr_option": {
            "value": ['suggest', 'float', [1e-6, 1e-3], {'log': True}],
            "reset": "y",
            "type": "cos"
        },
        "training_strategy": [
            {"mean": [-5, 5], "std": [1, 5]},
            {"mean": [-5, 5], "std": [0.2, 5]}
        ],
        "n_iter": 4,  # pour optuna, on réduit un peu pour tester
        "NDataT": 5000,
        "NDataV": 100,
        "period_checkpoint": -1,
        "script": ['Inter', 'Linearisation', 'Trainer.py'],
        "n_trials": 10,
        "prune": True,
        "retake_job": 11512632
    }

    import sys

    job_id = os.getenv("SLURM_JOB_ID", str(uuid.uuid4().hex))  # fallback si tu testes en local

    try:
        json_file = sys.argv[1]
        with open(json_file, "r") as f:
            temp_param = json.load(f)

        if temp_param['retake_job']:
            job_id = temp_param['retake_job']
            json_file = os.path.join(local, "Optuna", "Save", f"job_{job_id}", f"params_{job_id}.json")
            with open(json_file, "r") as f:
                temp_param = json.load(f)

        params.update(temp_param)

    except Exception as e:
        if len(sys.argv) == 0:
            print('nothing loaded')
        else:
            print(e)

    # Définir le répertoire de sauvegarde global
    run_dir = os.path.join(local, "Optuna", "Save", f"job_{job_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Définir le chemin de la base de données dans ce dossier
    db_path = os.path.join(run_dir, "optuna.db")
    storage = f"sqlite:///{db_path}"

    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        constant_liar=True,
        n_startup_trials=10
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=3,
        interval_steps=1
    )

    study = optuna.create_study(
        study_name="distributed-test",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner
    )

    # Boucle jusqu’à atteindre le nombre total souhaité
    while True:
        n_done = len([t for t in study.trials if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]])
        if n_done >= params['n_trials']:
            print(f"Objectif atteint : {n_done}/{params['n_trials']} trials terminés.")
            break

        study.optimize(lambda x: objective(x, run_dir, params), n_trials=1)


