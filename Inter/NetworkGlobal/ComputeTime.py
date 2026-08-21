from Inter.NetworkGlobal.Network import TransformerTranslator
import torch
import psutil, sys, os
import time
import tqdm

p = psutil.Process(os.getpid())

if sys.platform == "win32":
    p.nice(psutil.HIGH_PRIORITY_CLASS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_vitesse(batch_size=50, len_in=100):
    # Initialisation du modèle
    N = TransformerTranslator(10, 11, d_att=256, n_encoders=11, n_decoders=3, widths_embedding=[40], width_FF=[256],
                              n_heads=16, len_in=len_in, len_out=len_in)
    N.to(device)

    X = torch.normal(0, 1, (batch_size, len_in, 10), device=device)

    # --- 1. Phase de préchauffage (Warm-up) ---
    print("Préchauffage du GPU...")
    y_warmup = torch.zeros(batch_size, 0, 11).to(device)
    with torch.no_grad():
        for k in range(5):
            y_warmup = N.step(X, y_warmup, offset=k)
            N.reset_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # --- 2. Phase de Benchmarking lissée ---
    n_repetitions = 10
    print(f"Demarrage du benchmark (Repetitions: {n_repetitions}, Batch size: {batch_size}, Longueur cible: {len_in})...")

    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in tqdm.tqdm(range(n_repetitions)):
            # Réinitialisation de l'état pour chaque nouvelle évaluation du scénario
            y_list = []
            y = torch.zeros(batch_size, 0, 11).to(device)

            # Boucle d'inférence auto-régressive
            for k in range(len_in):
                y = N.step(X, y, offset=k)
                y_list.append(y)
            N.reset_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    # --- 3. Calcul des métriques moyennes ---
    total_time = end_time - start_time
    avg_time_per_run = total_time / n_repetitions

    total_pdw_interceptes_per_run = batch_size * len_in
    total_pdw_incidents_per_run = batch_size * len_in

    print("\n--- Resultats Moyennés ---")
    print(f"Temps moyen d'execution de la boucle : {avg_time_per_run:.4f} secondes.")
    print(f"Scenarios simultanes (Batch size)    : {batch_size}")
    print(
        f"Vitesse : {total_pdw_incidents_per_run / avg_time_per_run:.2f} PDW incidents/seconde.")

if __name__ == '__main__':
    len_in = 90
    batch_size = 50
    while batch_size < 200000:
        test_vitesse(batch_size=batch_size, len_in=len_in)
        batch_size = int(batch_size * 1.2)