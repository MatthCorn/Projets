import optuna
import subprocess
import json
import os
import uuid
import time

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index("Projets"))], "Projets")
TRAINER_SCRIPT = os.path.join(local, "Inter", "NetworkGlobal", "Trainer.py")


def objective(trial, RUN_DIR):
    # Hyperparamètres que l’on veut optimiser
    params = {
        "n_encoder": trial.suggest_int("n_encoder", 2, 4),
        "n_decoder": trial.suggest_int("n_decoder", 2, 4),
        "d_att": trial.suggest_categorical("d_att", [64, 128, ]),
        "widths_embedding": [trial.suggest_categorical("widths_embedding", [16, 32])],
        "width_FF": [trial.suggest_categorical("width_FF", [128])],
        "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "lr_option": {
            "value": trial.suggest_loguniform("lr", 1e-6, 1e-3),
            "reset": "y",
            "type": "cos"
        },
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-2),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1000]),
        "n_iter": 10,  # pour optuna, on réduit un peu pour tester
        "NDataT": 5000,
        "NDataV": 100,
        "period_checkpoint": -1
    }

    # Nom unique pour le fichier JSON
    json_file = os.path.join(RUN_DIR, f"params_{uuid.uuid4().hex}.json")
    with open(json_file, "w") as f:
        json.dump(params, f)

    # Fichier temporaire pour la progression
    progress_file = os.path.join(RUN_DIR, f"progress_{uuid.uuid4().hex}.txt")

    # Lancement du script d'entraînement
    process = subprocess.Popen(
        ["python", TRAINER_SCRIPT, json_file, progress_file],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    # Boucle de surveillance
    while process.poll() is None:
        time.sleep(10)  # toutes les 10 secondes
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

    # Nettoyage
    stdout, _ = process.communicate()
    os.remove(json_file)
    if os.path.exists(progress_file):
        os.remove(progress_file)

    # Lecture du score final
    score = float("inf")
    for line in stdout.splitlines():
        if "Final Error:" in line:
            score = float(line.split()[-1])

    trial.set_user_attr("message", stdout)
    return score

if __name__ == "__main__":
    job_id = os.getenv("SLURM_JOB_ID", str(uuid.uuid4().hex))  # fallback si tu testes en local

    # Définir le répertoire de sauvegarde global
    run_dir = os.path.join(local, "Optuna", "Save", f"job_{job_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Définir le chemin de la base de données dans ce dossier
    db_path = os.path.join(run_dir, "optuna.db")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="distributed-test",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner()
    )

    def my_callback(study, trial):
        print(f"{trial.user_attrs.get('message')}")

    study.optimize(lambda x: objective(x, run_dir), n_trials=5, callbacks=[my_callback])

