from Inter.Model.Scenario import Simulator, BiasedSimulator, FreqBiasedSimulator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import math


def value_to_rgb(value, min_val=0, max_val=2, colormap='plasma'):
    # Normalize the value between 0 and 1
    normalized_value = (value - min_val) / (max_val - min_val)
    # Clip the normalized value to ensure it stays within [0, 1]
    normalized_value = np.clip(normalized_value, 0, 1)
    # Get the colormap
    cmap = plt.get_cmap(colormap)
    # Map the normalized value to an RGB color
    rgb = cmap(normalized_value)  # Returns an RGBA tuple
    return rgb


if __name__ == '__main__':
    # --- CONFIGURATION ET SIMULATION ---
    df = 0.1
    N_total = 30
    n = 10
    range_plot = N_total + n
    dim = 10

    # S = Simulator(n, N_total, dim, sensitivity=df, seed=None)
    # S = BiasedSimulator(1, 3, n, N_total, dim, sensitivity=df, seed=None)
    S = FreqBiasedSimulator(1.2, 1, n, N_total, dim, sensitivity=df, seed=None)
    S.run()

    # --- AFFICHAGE 2D (STYLE SCRIPT 2) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    # 1. Plot des impulsions d'entrée (S.L)
    for i in range(len(S.L)):
        # Mapping des données selon la structure du simulateur
        T1 = i
        duration = S.L[i][-1]
        T2 = T1 + duration
        F = S.L[i][0]  # Fréquence centrale

        # Calcul de la valeur normalisée (comme dans la partie 3D ou le script 2)
        # S.L[i][1] est l'amplitude brute, on la passe dans tanh pour la couleur
        val_norm = 0.5 * math.tanh(S.L[i][1]) + 1

        r, g, b, a = value_to_rgb(val_norm)

        # Création du Rectangle (x, y, width, height)
        # Note: on centre le rectangle sur F, avec une hauteur de 2*df
        rect = Rectangle((T1, F - df),  # coin bas gauche (x, y)
                         duration,  # largeur
                         2 * df,  # hauteur
                         facecolor=(r, g, b, 0.8),
                         edgecolor='k',
                         linewidth=0.3)
        ax1.add_patch(rect)

    # 2. Plot des impulsions de sortie (Sensor Simulator R)
    R = S.sensor_simulator.R
    for i in range(len(R)):
        # Mapping des données (Attention aux index qui diffèrent légèrement entre Input et Output)
        T1 = i - R[i][-1]
        duration = R[i][-2]  # Dans S.sensor_simulator, la durée est souvent à -2
        T2 = T1 + duration
        F = R[i][0]

        val_norm = 0.5 * math.tanh(R[i][1]) + 1
        r, g, b, a = value_to_rgb(val_norm)

        rect = Rectangle((T1, F - df),
                         duration,
                         2 * df,
                         facecolor=(r, g, b, 0.8),
                         edgecolor='k',
                         linewidth=0.3)
        ax2.add_patch(rect)

    # Configuration des axes 2D
    ax1.set_xlim(-2, range_plot)
    ax1.set_ylim(0, 2)  # A adapter selon vos fréquences max/min
    ax1.set_ylabel('Fréquence')
    ax1.set_title("Impulsions d'entrée")

    ax2.set_xlim(-2, range_plot)
    ax2.set_ylim(0, 2)
    ax2.set_ylabel('Fréquence')
    ax2.set_xlabel('Temps')
    ax2.set_title("Impulsions de sortie")

    # Ajout des colorbars (comme dans le script 2)
    cmap = plt.get_cmap('plasma')
    norm = colors.Normalize(vmin=0, vmax=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for ax in (ax1, ax2):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sm, cax=cax)

    plt.tight_layout()
    plt.show()

    # --- AFFICHAGE 3D (Similaire au script original mais nettoyé) ---
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot 3D Input
    for i in range(len(S.L)):
        T1 = i
        T2 = T1 + S.L[i][-1]
        F = S.L[i][0]
        N = 0.5 * math.tanh(S.L[i][1]) + 1

        sommets = [
            [T1, F, N], [T2, F, N],
            [T1, F - df, 0], [T1, F + df, 0],
            [T2, F + df, 0], [T2, F - df, 0]
        ]
        surf = np.array([[sommets[0], sommets[0], sommets[0], sommets[0]],
                         [sommets[0], sommets[2], sommets[3], sommets[0]],
                         [sommets[1], sommets[5], sommets[4], sommets[1]],
                         [sommets[1], sommets[1], sommets[1], sommets[1]]])
        r, g, b, a = value_to_rgb(N)
        ax1.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))

    # Plot 3D Output
    R = S.sensor_simulator.R
    for i in range(len(R)):
        T1 = i - R[i][-1]
        T2 = T1 + R[i][-2]
        F = R[i][0]
        N = 0.5 * math.tanh(R[i][1]) + 1

        sommets = [
            [T1, F, N], [T2, F, N],
            [T1, F - df, 0], [T1, F + df, 0],
            [T2, F + df, 0], [T2, F - df, 0]
        ]
        surf = np.array([[sommets[0], sommets[0], sommets[0], sommets[0]],
                         [sommets[0], sommets[2], sommets[3], sommets[0]],
                         [sommets[1], sommets[5], sommets[4], sommets[1]],
                         [sommets[1], sommets[1], sommets[1], sommets[1]]])
        r, g, b, a = value_to_rgb(N)
        ax2.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))

    # Configuration propre des axes 3D
    for ax, title in zip([ax1, ax2], ["Impulsions d'entrée", "Impulsions de sortie"]):
        ax.set_xlim(-2, range_plot)
        ax.set_ylim(-3, 3)  # A ajuster selon l'échelle fréquentielle
        ax.set_zlim(0, 2)
        ax.set_box_aspect([2, 2, 0.5])

        # Suppression des ticks Z pour propreté
        ax.zaxis.set_ticks([])
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Pour les versions récentes de matplotlib

        ax.set_xlabel('Temps')
        ax.set_ylabel('Fréquence')
        ax.set_title(title)

        # Vue Ortho "Top Down"
        ax.set_proj_type('ortho')
        ax.view_init(elev=90, azim=-90)

        # Ajout Colorbar 3D
        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        fig.colorbar(sm, cax=cax)

    plt.show()