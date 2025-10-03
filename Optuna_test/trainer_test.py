import sys
import json
import random
import time

if __name__ == "__main__":
    json_file = sys.argv[1]
    with open(json_file, "r") as f:
        params = json.load(f)

    import torch, os
    gpu_id = torch.cuda.current_device()
    print(f"[Worker Node {os.getenv('SLURM_NODEID')} | GPU {gpu_id}] Starting training", flush=True)

    x = params["x"]
    y = params["y"]

    # Simulation d’un entraînement qui prend un peu de temps
    time.sleep(random.uniform(0.5, 2.0))

    # Objectif : minimum en (x=2, y=-3)
    error = (x - 2) ** 2 + (y + 3) ** 2

    # Ce print est ce que Optuna va chercher dans stdout
    print(f"Final Error: {error}", flush=True)
