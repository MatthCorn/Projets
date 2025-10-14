import optuna, sys, os

if __name__ == '__main__':
    # Récupère le SLURM_JOB_ID (défini automatiquement dans l'environnement du job)
    job_id = os.environ.get("SLURM_JOB_ID", "nojobid")

    # Base du projet
    local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index("Projets"))], "Projets")

    # Dossier de sauvegarde
    SAVE_ROOT = os.path.join(local, "Optuna", "Save")

    # Dossier spécifique à ce job SLURM
    RUN_DIR = os.path.join(SAVE_ROOT, f"job_{job_id}")
    os.makedirs(RUN_DIR, exist_ok=True)

    # Copie du chemin de la base à l'intérieur du dossier
    db_path = os.path.join(RUN_DIR, "optuna.db")
    storage = f"sqlite:///{db_path}"

    # Crée la base et l'étude si elles n'existent pas
    study = optuna.create_study(
        study_name="distributed-test",
        storage=storage,
        load_if_exists=True,
        direction="minimize"
    )

    print("Base de données Optuna initialisée : optuna.db")
