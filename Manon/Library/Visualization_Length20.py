import os, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

################################################################################################################################################
def plot_visuals(InputData, OutputData, PredictionData, filename, save_path):
   
    def value_to_rgb(v):
        cmap = plt.get_cmap('plasma')
        rgba = cmap(v / 2.0)  # normalize to [0,1]
        return rgba[0], rgba[1], rgba[2], rgba[3]

    # INPUTS
    df = 0.2
    S = InputData
    R = OutputData
    L = PredictionData

    # DEBUG
    if len(PredictionData) > 0:
        freq_preds = [p[0] for p in PredictionData]
        print(f"[DEBUG] Predicted Frequencies: min={min(freq_preds):.2f}, max={max(freq_preds):.2f}")

    # derive limits from sequence length and data :
    ################################################################################################################################################
    len_in = 20                    # number of input pulses
    t_max = len_in + 5                    # margin on the time axis
    ################################################################################################################################################
    
    # collect frequency range across S, R, L
    all_freqs = []
    all_freqs += [x[0] for x in S] if len(S) else []
    all_freqs += [x[0] for x in R] if len(R) else []
    all_freqs += [x[0] for x in L] if len(L) else []
    if all_freqs:
        fmin, fmax = float(min(all_freqs)), float(max(all_freqs))
        # add a small visual margin based on df and data spread
        margin = max(df, 0.05*(fmax - fmin + 1.0))
        y_lo, y_hi = fmin - margin, fmax + margin
    else:
        y_lo, y_hi = -3, 3

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(18, 6))

    # --------- INPUT surfaces ----------
    for i in range(len(S)):
        T1 = i
        T2 = T1 + S[i][-1]
        F  = S[i][0]
        N  = 0.5 * math.tanh(S[i][1]) + 1
        sommets = [
            [T1, F, N],
            [T2, F, N],
            [T1, F-df, 0],
            [T1, F+df, 0],
            [T2, F+df, 0],
            [T2, F-df, 0]
        ]
        surf = np.array([[sommets[0], sommets[0], sommets[0], sommets[0]],
                         [sommets[0], sommets[2], sommets[3], sommets[0]],
                         [sommets[1], sommets[5], sommets[4], sommets[1]],
                         [sommets[1], sommets[1], sommets[1], sommets[1]]])
        r, g, b, a = value_to_rgb(N)
        ax1.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))

    # --------- OUTPUT surfaces ----------
    for i in range(len(R)):
        T1 = i - R[i][-1]
        T2 = T1 + R[i][-2]
        F  = R[i][0]
        N  = 0.5 * math.tanh(R[i][1]) + 1
        sommets = [
            [T1, F, N],
            [T2, F, N],
            [T1, F-df, 0],
            [T1, F+df, 0],
            [T2, F+df, 0],
            [T2, F-df, 0]
        ]
        surf = np.array([[sommets[0], sommets[0], sommets[0], sommets[0]],
                         [sommets[0], sommets[2], sommets[3], sommets[0]],
                         [sommets[1], sommets[5], sommets[4], sommets[1]],
                         [sommets[1], sommets[1], sommets[1], sommets[1]]])
        r, g, b, a = value_to_rgb(N)
        ax2.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))

    # --------- PRED surfaces ----------
    for i in range(len(L)):
        T1 = i - L[i][-1]
        T2 = T1 + L[i][-2]
        F  = L[i][0]
        N  = 0.5 * math.tanh(L[i][1]) + 1
        sommets = [
            [T1, F, N],
            [T2, F, N],
            [T1, F-df, 0],
            [T1, F+df, 0],
            [T2, F+df, 0],
            [T2, F-df, 0]
        ]
        surf = np.array([[sommets[0], sommets[0], sommets[0], sommets[0]],
                         [sommets[0], sommets[2], sommets[3], sommets[0]],
                         [sommets[1], sommets[5], sommets[4], sommets[1]],
                         [sommets[1], sommets[1], sommets[1], sommets[1]]])
        r, g, b, a = value_to_rgb(N)
        ax3.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))

    # colorbars
    cmap = plt.get_cmap('plasma')
    norm = colors.Normalize(vmin=0, vmax=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, shrink=0.4)
    fig.colorbar(sm, ax=ax2, shrink=0.4)
    fig.colorbar(sm, ax=ax3, shrink=0.4)

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(-2, t_max)       # time axis scaled by len_in (+5 margin)
        ax.set_ylim(y_lo, y_hi)      # frequency axis from data with margin
        ax.set_zlim(0, 2)
        ax.set_box_aspect([2, 2, 0.5])
        ax.zaxis.set_ticks([])
        ax.zaxis.set_ticklabels([])
        ax.zaxis.label.set_visible(False)
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.set_proj_type('ortho')
        ax.view_init(elev=90, azim=-90)

    ax1.set_xlabel('temps');      ax1.set_ylabel('fréquence'); ax1.set_title("Impulsions d'entrée")
    ax2.set_xlabel('temps');      ax2.set_ylabel('fréquence'); ax2.set_title("Impulsions de sortie")
    ax3.set_xlabel('temps');      ax3.set_ylabel('fréquence'); ax3.set_title("Prédictions d'impulsions")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename))
    plt.close()
################################################################################################################################################

