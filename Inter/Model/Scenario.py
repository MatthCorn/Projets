import numpy as np
from Inter.Model.Sensor import Simulator as SensorSimulator

class Simulator:
    def __init__(self, n, N, dim, sensitivity=0.2, seed=None):
        self.n = n  # Nombre de vecteurs présents simultanément
        self.N = N  # Nombre total de vecteurs dans le scénario
        self.dim = dim
        self.T = 0
        self.V = [] # contient tous les vecteurs présents simultanément à un instant donné
        self.A = [] # contient l'âge de chaque vecteur présent à un instant donné
        self.L = [] # contient chaque vecteur du scénario ainsi que leurs ages à leurs disparitions

        self.sensor_simulator = SensorSimulator(dim=dim, sensitivity=sensitivity)
        if seed is not None:
            np.random.seed(seed)

    def Step(self):
        v = list(np.random.normal(0, 1, self.dim))

        if self.T >= self.n:
            # sélection d'un indice dans les vecteurs du palier, correspondant au vecteur disparaissant
            m = len(self.V)
            p = [(1+a)/(m+sum(np.array(self.A))) for a in self.A]
            k = np.random.choice(range(m), p=p)

            # on supprime le vecteur disparu de self.V, et on retient son âge depuis self.A
            a_out, v_out = self.A.pop(k), self.V.pop(k)

            # on inscrit l'âge du vecteur juste disparu dans son instance présente dans self.L
            i = self.L.index(v_out)
            self.L[i] += [a_out]

        if self.T < self.N:
            # on ajoute le nouveau vecteur dans self.L et self.V, ainsi que son âge actuel dans self.A
            self.L.append(v)
            self.V.append(v)
            self.A.append(0)

        self.T += 1
        for i in range(len(self.A)):
            self.A[i] += 1

    def Run(self):
        while self.sensor_simulator.running:
            if (self.T == 0) or (len(self.V) > 0):
                self.Step()
            self.sensor_simulator.Process(self.V)



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
    df = 0.2
    S = Simulator(5, 30, 4, sensitivity=df, seed=None)
    S.Run()

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

    ax1.set_xlim(-2, 35)
    ax1.set_ylim(0, 2)
    ax2.set_xlim(-2, 35)
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

    ax1.set_xlim(-2, 35)
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
    ax2.set_xlim(-2, 35)
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