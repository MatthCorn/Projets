import optuna, sys

if __name__ == '__main__':

    db_path = sys.argv[1]
    storage = f"sqlite:///{db_path}"

    # Crée la base et l'étude si elles n'existent pas
    study = optuna.create_study(
        study_name="distributed-test",
        storage=storage,
        load_if_exists=True,
        direction="minimize"
    )

    print("Base de données Optuna initialisée : optuna.db")
