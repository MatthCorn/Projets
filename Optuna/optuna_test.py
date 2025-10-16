import optuna
import subprocess
import json
import os
import sys
import uuid
import random
import time

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index("Projets"))], "Projets")
TRAINER_SCRIPT = os.path.join(local, "Optuna", "trainer_test.py")

def objective(trial):
    # Hyperparamètres à tester (jouet mais en JSON comme ton vrai cas)
    params = {
        "x": trial.suggest_float("x", -10, 10),
        "y": trial.suggest_float("y", -10, 10)
    }

    # Nom unique pour le fichier JSON
    json_file = f"params_{uuid.uuid4().hex}.json"
    with open(json_file, "w") as f:
        json.dump(params, f)

    # Appel du script externe (simulateur d'entraînement)
    result = subprocess.run(
        [sys.executable, TRAINER_SCRIPT, json_file],
        capture_output=True,
        text=True
    )

    # Nettoyage
    os.remove(json_file)

    # Récupérer le score imprimé par trainer_test.py
    score = float("inf")
    for line in result.stdout.splitlines():
        if "Final Error:" in line:
            score = float(line.split()[-1])

    trial.set_user_attr('message',
                        "Current working dir:" + str(os.getcwd()) + '\n' +
                        "trainer_test.py exists:" + str(os.path.exists(TRAINER_SCRIPT)) + '\n' +
                        "Python executable:" + str(sys.executable)
    )
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
        direction="minimize"
    )

    def my_callback(study, trial):
        print(f"{trial.user_attrs.get('message')}")

    study.optimize(objective, n_trials=5, callbacks=[my_callback])

