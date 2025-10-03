import optuna
import subprocess
import json
import os
import sys
import uuid
import random
import time

def objective(trial):
    import torch, os
    print(
        f"[Worker {os.getenv('SLURM_PROCID')} | Node {os.getenv('SLURM_NODEID')}] Trial {trial.number} running on GPU {torch.cuda.current_device()}")

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
        ["python", os.path.join("Projets", "Optuna_test", "trainer_test.py"), json_file],
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

    return score


if __name__ == "__main__":
    db_path = sys.argv[1]
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="distributed-test",
        storage=storage,
        load_if_exists=True,
        direction="minimize"
    )
    study.optimize(objective, n_trials=50)
