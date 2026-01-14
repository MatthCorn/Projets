from Inter.Model.Scenario import Simulator, BiasedSimulator, FreqBiasedSimulator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math


def value_to_rgb(value, min_val=0, max_val=2, colormap='plasma'):
    # Normalize the value between 0 and 1
    normalized_value = (value - min_val) / (max_val - min_val)
    normalized_value = np.clip(normalized_value, 0, 1)
    cmap = plt.get_cmap(colormap)
    rgb = cmap(normalized_value)
    return rgb


if __name__ == '__main__':
    # --- CONFIGURATION ET SIMULATION ---
    df = 0.1
    N_total = 30
    n = 10
    range_plot = N_total + n
    dim = 10

    # S = Simulator(n, N_total, dim, sensitivity=df, seed=None)
    S = FreqBiasedSimulator(1.2, 1, n, N_total, dim, sensitivity=df, seed=None)
    S.run()

    # --- INITIALISATION FIGURE (1 ligne, 2 colonnes, AXES LIÉS) ---
    # sharex=True, sharey=True : lier le zoom et le pan des deux figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # Variables pour calculer l'échelle Y automatiquement
    y_min_data = float('inf')
    y_max_data = float('-inf')

    # --- 1. PLOT ENTREE (GAUCHE) ---
    for i in range(len(S.L)):
        T1 = i
        duration = S.L[i][-1]
        F = S.L[i][0]

        # Mise à jour min/max
        y_min_data = min(y_min_data, F)
        y_max_data = max(y_max_data, F)

        val_norm = 0.5 * math.tanh(S.L[i][1]) + 1
        r, g, b, a = value_to_rgb(val_norm)

        rect = Rectangle((T1, F - df),
                         duration,
                         2 * df,
                         facecolor=(r, g, b, 0.8),
                         edgecolor='k',
                         linewidth=0.3)
        ax1.add_patch(rect)

    # --- 2. PLOT SORTIE (DROITE) ---
    R = S.sensor_simulator.R
    for i in range(len(R)):
        T1 = i - R[i][-1]
        duration = R[i][-2]
        F = R[i][0]

        # Mise à jour min/max
        y_min_data = min(y_min_data, F)
        y_max_data = max(y_max_data, F)

        val_norm = 0.5 * math.tanh(R[i][1]) + 1
        r, g, b, a = value_to_rgb(val_norm)

        rect = Rectangle((T1, F - df),
                         duration,
                         2 * df,
                         facecolor=(r, g, b, 0.8),
                         edgecolor='k',
                         linewidth=0.3)
        ax2.add_patch(rect)

    # --- REGLAGE DES LIMITES ---
    margin_y = 5 * df

    # On applique les limites sur ax1, ax2 suivra automatiquement grâce à sharex/sharey
    ax1.set_xlim(-2, range_plot)
    ax1.set_ylim(y_min_data - margin_y, y_max_data + margin_y)

    # Labels et Titres
    ax1.set_xlabel('Temps')
    ax1.set_ylabel('Fréquence')
    ax1.set_title("Impulsions d'entrée")

    ax2.set_xlabel('Temps')
    # Pas besoin de set_ylabel sur ax2 car l'axe est partagé et les ticks sont masqués par défaut
    ax2.set_title("Impulsions de sortie")

    # --- COLORBARS ---
    cmap = plt.get_cmap('plasma')
    norm = colors.Normalize(vmin=0, vmax=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sm, cax=cax)

    plt.tight_layout()
    plt.show()