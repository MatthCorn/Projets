from Inter.Model.Sensor import Simulator as SensorSimulator
from Inter.Model.Scenario import FreqBiasedSimulatorTemplate, Simulator
from copy import deepcopy

class ResettingSimulator(SensorSimulator):
    def __init__(self, reset_period, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_period = reset_period
        self.P2 = []     # contient les vecteurs de suivis, qui enregistrent itérativement les informations des vecteurs présents
        self.TM2 = []    # contient le temps de dernière mise-à-jour de chaque vecteur de suivi
        self.TI2 = []    # contient le temps d'apparition de chaque vecteur de suivi
        self.R2 = []     # contient tous les vecteurs interceptés, ajoutés à la fin de leurs interceptions
        self.T2 = 0
        self.running2 = True
        
    def Process(self, Input):
        # échange des variables et traitement miroire
        T, P, TM, TI, R, running = self.T, self.P, self.TM, self.TI, self.R, self.running
        self.T, self.P, self.TM, self.TI, self.R, self.running = self.T2, self.P2, self.TM2, self.TI2, self.R2, self.running2
        
        if not (self.T % self.reset_period):
            self.P = []  # reset des vecteurs de suivis, qui enregistrent itérativement les informations des vecteurs présents
            self.TM = []  # reset des temps de dernière mise-à-jour de chaque vecteur de suivi
            self.TI = []  # reset des temps d'apparition de chaque vecteur de suivi
        
        super().Process(Input)
        
        # reéchange des variables et traitement standard
        self.T2, self.P2, self.TM2, self.TI2, self.R2, self.running2 = self.T, self.P, self.TM, self.TI, self.R, self.running
        self.T, self.P, self.TM, self.TI, self.R, self.running = T, P, TM, TI, R, running
        
        super().Process(Input)
class ScenarioSimulator(Simulator):
    def __init__(self, reset_period, n, N, dim, n_sat=5, n_mes=100, sensitivity=0.2, seed=None, WeightF=None, WeightL=None, model_path=None):
        super().__init__(n, N, dim, n_sat=n_sat, n_mes=n_mes, sensitivity=sensitivity, seed=seed, WeightF=WeightF, WeightL=WeightL, model_path=model_path)
        self.sensor_simulator = ResettingSimulator(reset_period, dim=dim, sensitivity=sensitivity, n_sat=n_sat, n_mes=n_mes, WeightF=self.weight_f, WeightL=self.weight_l)

class FreqBiasedSimulator(FreqBiasedSimulatorTemplate, ScenarioSimulator):
    pass

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
    N_total = 500
    n = 6
    range_plot = N_total + n
    dim = 10

    # S = Simulator(n, N_total, dim, sensitivity=df, seed=None)
    S = FreqBiasedSimulator(0.5, 1, 100, n, N_total, dim, sensitivity=df, seed=None)
    S.run()

    # --- INITIALISATION FIGURE (1 ligne, 2 colonnes, AXES LIÉS) ---
    # sharex=True, sharey=True : lier le zoom et le pan des deux figures
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)

    # Variables pour calculer l'échelle Y automatiquement
    y_min_data = float('inf')
    y_max_data = float('-inf')

    # # --- 1. PLOT ENTREE (GAUCHE) ---
    # for i in range(len(S.L)):
    #     T1 = i
    #     duration = S.L[i][-1]
    #     F = S.L[i][0]
    #
    #     # Mise à jour min/max
    #     y_min_data = min(y_min_data, F)
    #     y_max_data = max(y_max_data, F)
    #
    #     val_norm = 0.5 * math.tanh(S.L[i][1]) + 1
    #     r, g, b, a = value_to_rgb(val_norm)
    #
    #     rect = Rectangle((T1, F - df),
    #                      duration,
    #                      2 * df,
    #                      facecolor=(r, g, b, 0.8),
    #                      edgecolor='k',
    #                      linewidth=0.3)
    #     ax1.add_patch(rect)

    # --- 2. PLOT SORTIE (Gauche) ---
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
        ax1.add_patch(rect)

    R1 = R

    # --- 3. PLOT SORTIE avec reset (Milieu) ---
    R = S.sensor_simulator.R2
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
    R2 = R

    # --- 3. PLOT DIFFERENCE SYMETRIQUE DES UNIONS (DROITE) ---
    from collections import defaultdict
    def get_sig(i, r):
        # Arrondi pour éviter les erreurs de flottants
        return (round(i - r[-1], 5), round(r[0], 5), round(r[-2], 5))


    dict_R1 = {get_sig(i, r): r for i, r in enumerate(R1)}
    dict_R2 = {get_sig(i, r): r for i, r in enumerate(R2)}

    set_R1 = set(dict_R1.keys())
    set_R2 = set(dict_R2.keys())

    # 1. Optimisation : Différence symétrique des ensembles discrets
    only_R1_sigs = set_R1 - set_R2
    only_R2_sigs = set_R2 - set_R1


    # 2. Conversion en intervalles [début, fin] groupés par Fréquence
    def build_intervals(sig_set):
        intervals_by_F = defaultdict(list)
        for sig in sig_set:
            T1, F, duration = sig
            intervals_by_F[F].append([T1, T1 + duration])
        return intervals_by_F


    intervals_R1 = build_intervals(only_R1_sigs)
    intervals_R2 = build_intervals(only_R2_sigs)

    # ou dans le cas où il y a plein de petites erreurs de positionnement (traitement bien plus long)
    # intervals_R1 = build_intervals(set_R1)
    # intervals_R2 = build_intervals(set_R2)

    # Toutes les fréquences concernées par des différences
    all_F = set(intervals_R1.keys()).union(set(intervals_R2.keys()))


    # 3. Fonction pour fusionner les intervalles (Calcul de l'Union)
    def merge_intervals(intervals):
        if not intervals: return []
        # Tri par temps de départ
        intervals.sort(key=lambda x: x[0])
        merged = [list(intervals[0])]
        for current in intervals[1:]:
            prev = merged[-1]
            # Si chevauchement ou contiguïté parfaite
            if current[0] <= prev[1]:
                prev[1] = max(prev[1], current[1])
            else:
                merged.append(list(current))
        return merged


    # 4. Fonction pour la soustraction géométrique (A \ B)
    def subtract_intervals(A, B):
        result = []
        for a in A:
            current_pieces = [a]
            for b in B:
                next_pieces = []
                for p in current_pieces:
                    # Aucun chevauchement
                    if b[1] <= p[0] or b[0] >= p[1]:
                        next_pieces.append(p)
                    else:
                        # Si b coupe p, on conserve les morceaux restants de p
                        if p[0] < b[0]:
                            next_pieces.append([p[0], b[0]])
                        if p[1] > b[1]:
                            next_pieces.append([b[1], p[1]])
                current_pieces = next_pieces
            result.extend(current_pieces)
        return result


    # 5. Calcul de la différence symétrique et tracé
    for F in all_F:
        U1 = merge_intervals(intervals_R1.get(F, []))
        U2 = merge_intervals(intervals_R2.get(F, []))

        # (U1 \ U2) U (U2 \ U1)
        diff_1 = subtract_intervals(U1, U2)
        diff_2 = subtract_intervals(U2, U1)

        # Concaténation des deux listes (elles sont disjointes par nature)
        sym_diff = diff_1 + diff_2

        for segment in sym_diff:
            T_start = segment[0]
            duration = segment[1] - segment[0]

            # Mise à jour min/max pour l'échelle Y
            y_min_data = min(y_min_data, F)
            y_max_data = max(y_max_data, F)

            # Affichage en gris
            rect = Rectangle((T_start, F - df), duration, 2 * df,
                             facecolor=(0.5, 0.5, 0.5, 0.8),  # Gris
                             edgecolor='k', linewidth=0.3)
            ax3.add_patch(rect)

    ax3.set_title("Différence symétrique géométrique")

    # --- REGLAGE DES LIMITES ---
    margin_y = 5 * df

    # On applique les limites sur ax1, ax2 suivra automatiquement grâce à sharex/sharey
    ax1.set_xlim(-2, range_plot)
    ax1.set_ylim(y_min_data - margin_y, y_max_data + margin_y)

    # Labels et Titres
    ax1.set_xlabel('Temps')
    ax1.set_ylabel('Fréquence')
    ax1.set_title("Impulsions de sortie")

    ax2.set_xlabel('Temps')
    # Pas besoin de set_ylabel sur ax2 car l'axe est partagé et les ticks sont masqués par défaut
    ax2.set_title("Impulsions de sortie avec reset périodique")

    ax3.set_xlabel('Temps')
    # Pas besoin de set_ylabel sur ax2 car l'axe est partagé et les ticks sont masqués par défaut
    ax3.set_title('Différence "sans" vs "avec" reset')

    # --- COLORBARS ---
    cmap = plt.get_cmap('plasma')
    norm = colors.Normalize(vmin=0, vmax=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sm, cax=cax)

    plt.tight_layout()
    plt.show()