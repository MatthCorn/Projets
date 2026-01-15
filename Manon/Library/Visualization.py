import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import colors
import numpy as np
import random
import math
import os

################################################################################################################################################
def value_to_rgb(value, min_val=0, max_val=2, colormap='plasma'):
        # Normalize the value between 0 and 1
        normalized_value = (value - min_val) / (max_val - min_val)

        # Clip the normalized value to ensure it stays within [0, 1]
        normalized_value = np.clip(normalized_value, 0, 1)

        # Get the colormap
        cmap = plt.get_cmap(colormap)

        # Map the normalized value to an RGB color
        rgb = cmap(normalized_value)  # Returns an RGBA tuple, we need the RGB part

        return rgb
################################################################################################################################################

################################################################################################################################################
def plot_visuals(InputData, OutputData, PredictionData,filename, save_path) :

    # DATE GENERATION : 
    ################################################################################################################################################
    # INPUT DATA : S.L = sequence d'entrée
    df = 0.2

   # => REVIEW DF / Sensitivity 

    # S = Simulator(5, 30, 4, sensitivity=df, seed=None)
    # S = BiasedSimulator(1, 3, 5, 30, 4, sensitivity=df, seed=None)
    # S = FreqBiasedSimulator(0.8, 1, 5, 30, 4, sensitivity=df, seed=None)
    #S.run()
    S = InputData
    #print("Input Data : ", S)
    #print("Input Data at Index 0 : ", S.L[0])
    
    # OUTPUT DATA : 
    # R = S.sensor_simulator.R # sequence de Sortie
    R = OutputData
    #print("Output Data : ", R)
    #print("Output Data at Index 0 : ", R[0])

    # Predictions : 
    L = PredictionData
    #print("Prediction Data : ", L)

    freq_preds = [p[0] for p in PredictionData]
    ################################################################################################################################################
    
    #fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(18, 6))

    for i in range(len(S)):
        T1 = i
        T2 = T1 + S[i][-1]
        F = S[i][0]
        N = 0.5 * math.tanh(S[i][1]) + 1
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

        # Plot the surface
        r, g, b, a = value_to_rgb(N)
        ax1.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))

    #R = S.sensor_simulator.R
    for i in range(len(R)):
        T1 = i - R[i][-1]
        T2 = T1 + R[i][-2]
        F = R[i][0]
        N = 0.5 * math.tanh(R[i][1]) + 1

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

        # Plot the surface
        r, g, b, a = value_to_rgb(N)
        ax2.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))


    #For Predictions 
    for i in range(len(L)):
        T1 = i - L[i][-1]
        T2 = T1 + L[i][-2]
        F = L[i][0]
        N = 0.5 * math.tanh(L[i][1]) + 1

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

        # Plot the surface
        r, g, b, a = value_to_rgb(N)
        ax3.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))


    # Define the colormap and normalization (vmin, vmax)
    cmap = plt.get_cmap('plasma')  # You can choose 'plasma', 'coolwarm', etc.
    norm = colors.Normalize(vmin=0, vmax=2)  # Define the range for the colorbar

    # Create a ScalarMappable to be used for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Optional, required in some cases to avoid warnings

    # Add the colorbar to the figure without any plot
    fig.colorbar(sm, ax=ax2, shrink=0.4)
    fig.colorbar(sm, ax=ax1, shrink=0.4)
    fig.colorbar(sm, ax=ax3, shrink=0.4)

    ax1.set_xlim(-2, 17)
    ax1.set_ylim(-3, 3)
    ax1.set_zlim(0, 2)
    ax1.set_box_aspect([2, 2, 0.5])
    ax1.zaxis.set_ticks([])  # Remove z-axis ticks
    ax1.zaxis.set_ticklabels([])  # Remove z-axis tick labels
    ax1.zaxis.label.set_visible(False)  # Hide z-axis label
    ax1.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax1.set_xlabel('temps')
    ax1.set_ylabel('fréquence')
    ax1.set_title("Impulsions d'entrée")
    ax1.set_proj_type('ortho')
    ax1.view_init(elev=90, azim=-90)

    ax2.set_xlim(-2, 17)
    ax2.set_ylim(-3, 3)
    ax2.set_zlim(0, 2)
    ax2.set_box_aspect([2, 2, 0.5])
    ax2.zaxis.set_ticks([])  # Remove z-axis ticks
    ax2.zaxis.set_ticklabels([])  # Remove z-axis tick labels
    ax2.zaxis.label.set_visible(False)  # Hide z-axis label
    ax2.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax2.set_xlabel('temps')
    ax2.set_ylabel('fréquence')
    ax2.set_title("Impulsions de sortie")
    ax2.set_proj_type('ortho')
    ax2.view_init(elev=90, azim=-90)

    ax3.set_xlim(-2, 17)
    ax3.set_ylim(-3, 3)
    ax3.set_zlim(0, 2)
    ax3.set_box_aspect([2, 2, 0.5])
    ax3.zaxis.set_ticks([])  # Remove z-axis ticks
    ax3.zaxis.set_ticklabels([])  # Remove z-axis tick labels
    ax3.zaxis.label.set_visible(False)  # Hide z-axis label
    ax3.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax3.set_xlabel('temps')
    ax3.set_ylabel('fréquence')
    ax3.set_title("Predictions Impulsions")
    ax3.set_proj_type('ortho')
    ax3.view_init(elev=90, azim=-90)

    # M1 - Save graph : 
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename))
    plt.close()

    # M2 - Only show graph :
    #plt.show()