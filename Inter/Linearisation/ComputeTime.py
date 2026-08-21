from Inter.Linearisation.GRU import GRUNetwork
import torch
import psutil, sys, os
import tqdm

p = psutil.Process(os.getpid())

if sys.platform == "win32":
    p.nice(psutil.HIGH_PRIORITY_CLASS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_vitesse(batch_size=50, len_in=50):
    # Initialisation du modèle GRU aux dimensions compatibles
    N = GRUNetwork(
        input_dim_1=10,
        input_dim_2=11,
        hidden_dim=640,
        output_dim=11,
        n_layers=5
    )
    N.to(device)
    N.eval()

    X1 = torch.normal(0, 1, (batch_size, len_in, 10), device=device)

    # Création du masque explicitement sur le GPU pour contourner
    # le bug de création CPU dans GRUNetwork.step()
    next_mask = torch.zeros(batch_size, 1, device=device)

    # --- 1. Phase de préchauffage (Warm-up) ---
    print(f"Préchauffage du GPU pour batch_size={batch_size}...")
    y_warmup = torch.zeros(batch_size, 11, device=device)
    with torch.no_grad():
        N.cache = None
        for k in range(5):
            y_warmup, _ = N.step(X1[:, k, :], y_warmup, next_mask)
            y_warmup = y_warmup.squeeze(1)  # Rétablissement de la dimension pour l'itération suivante

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # --- 2. Phase de Benchmarking lissée ---
    n_repetitions = 50
    print(
        f"Demarrage du benchmark (Repetitions: {n_repetitions}, Batch size: {batch_size}, Longueur cible: {len_in})...")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()
        for _ in tqdm.tqdm(range(n_repetitions)):
            y = torch.zeros(batch_size, 11, device=device)
            # Purge manuelle car N.reset_cache() n'existe pas dans GRUNetwork
            N.cache = None

            # Boucle d'inférence flux-à-flux
            for k in range(len_in):
                y, _ = N.step(X1[:, k, :], y, next_mask)
                y = y.squeeze(1)
        end_event.record()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # --- 3. Calcul des métriques moyennes ---
    total_time = start_event.elapsed_time(end_event) / 1000.0
    avg_time_per_run = total_time / n_repetitions

    total_pdw_incidents_per_run = batch_size * len_in

    print("\n--- Resultats Moyennés ---")
    print(f"Temps moyen d'execution de la boucle : {avg_time_per_run:.4f} secondes.")
    print(f"Scenarios simultanes (Batch size)    : {batch_size}")
    print(f"Vitesse : {total_pdw_incidents_per_run / avg_time_per_run:.2f} PDW incidents/seconde.\n")


if __name__ == '__main__':
    batch_size = 50
    len_in = 50  # On fixe la séquence temporelle pour évaluer le traitement de flux

    while batch_size <= 100000:
        test_vitesse(batch_size=batch_size, len_in=len_in)
        batch_size = int(batch_size * 1.5)