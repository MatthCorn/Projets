from Inter.Model.Scenario import Simulator, BiasedSimulator, FreqBiasedSimulator
import numpy as np

if __name__ == '__main__':
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

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import random
    import math
    df = 0.1
    N = 30
    n = 10
    range_plot = N + n
    dim = 10
    S = Simulator(n, N, dim, sensitivity=df, seed=None)
    # S = BiasedSimulator(1, 3, n, N, dim, sensitivity=df, seed=None)
    # S = FreqBiasedSimulator(1.2, 1, n, N, dim, sensitivity=df, seed=None)
    S.run()


    fig, (ax1, ax2) = plt.subplots(2)

    for i in range(len(S.L)):
        T1 = i
        T2 = T1 + S.L[i][-1]
        N = math.tanh(S.L[i][0]) + 1
        r, g, b = random.random(), random.random(), random.random()
        Rectangle = Polygon(((T1, 0), (T1, N), (T2, N), (T2, 0)), fc=(r, g, b, 0.1), ec=(0, 0, 0, 1), lw=2)
        ax1.add_artist(Rectangle)


    R = S.sensor_simulator.R
    for i in range(len(R)):
        T1 = i - R[i][-1]
        T2 = T1 + R[i][-2]
        N = math.tanh(R[i][0]) + 1
        r, g, b = random.random(), random.random(), random.random()
        Rectangle = Polygon(((T1, 0), (T1, N), (T2, N), (T2, 0)), fc=(r, g, b, 0.1), ec=(0, 0, 0, 1), lw=2)
        ax2.add_artist(Rectangle)

    ax1.set_xlim(-2, range_plot)
    ax1.set_ylim(0, 2)
    ax2.set_xlim(-2, range_plot)
    ax2.set_ylim(0, 2)
    plt.show()

    from matplotlib import colors
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

    for i in range(len(S.L)):
        T1 = i
        T2 = T1 + S.L[i][-1]
        F = S.L[i][0]
        N = 0.5 * math.tanh(S.L[i][1]) + 1
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



    R = S.sensor_simulator.R
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


    # Define the colormap and normalization (vmin, vmax)
    cmap = plt.get_cmap('plasma')  # You can choose 'plasma', 'coolwarm', etc.
    norm = colors.Normalize(vmin=0, vmax=2)  # Define the range for the colorbar

    # Create a ScalarMappable to be used for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Optional, required in some cases to avoid warnings

    # Add the colorbar to the figure without any plot
    fig.colorbar(sm, ax=ax2, shrink=0.4)
    fig.colorbar(sm, ax=ax1, shrink=0.4)

    ax1.set_xlim(-2, range_plot)
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
    ax2.set_xlim(-2, range_plot)
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
    plt.show()