import numpy as np
from Inter.Sensor import Simulator as SensorSimulator

class Simulator:
    def __init__(self, n, N, dim, sensitivity_fourier=0.2, sensitivity_sensor=0.1):
        self.n = n
        self.N = N
        self.dim = dim
        self.T = 0
        self.L = []
        self.V = []
        self.A = []
        self.sensor_simulator = SensorSimulator(dim=dim, sensitivity_fourier=sensitivity_fourier, sensitivity_sensor=sensitivity_sensor)

    def Step(self):
        v = list(np.random.normal(0, 1, self.dim))

        if self.T >= self.n:
            # sélection d'un indice dans les vecteurs du palier
            m = len(self.V)
            p = [(1+a)/(m+sum(np.array(self.A))) for a in self.A]
            k = np.random.choice(range(m), p=p)
            a_out, v_out = self.A.pop(k), self.V.pop(k)
            i = self.L.index(v_out)
            self.L[i] += [a_out]

        if self.T < self.N:
            self.L.append(v)
            self.V.append(v)
            self.A.append(0)

        self.T += 1
        for i in range(len(self.A)):
            self.A[i] += 1

        self.sensor_simulator.Process(self.V)

    def Run(self):
        while self.T == 0 or len(self.V) > 0:
            self.Step()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import random
    import math
    S = Simulator(5, 30, 4, sensitivity_fourier=0.1, sensitivity_sensor=0.05)
    S.Run()

    fig, (ax1, ax2) = plt.subplots(2)

    for i in range(len(S.L)):
        T1 = i
        T2 = T1 + S.L[i][-1]
        N = math.sqrt(abs(S.L[i][0]))
        r, g, b = random.random(), random.random(), random.random()
        Rectangle = Polygon(((T1, 0), (T1, N), (T2, N), (T2, 0)), fc=(r, g, b, 0.1), ec=(0, 0, 0, 1), lw=2)
        ax1.add_artist(Rectangle)


    R = S.sensor_simulator.R
    for i in range(len(R)):
        T1 = i - R[i][-1]
        T2 = T1 + R[i][-2]
        N = math.sqrt(abs(R[i][0]))
        r, g, b = random.random(), random.random(), random.random()
        Rectangle = Polygon(((T1, 0), (T1, N), (T2, N), (T2, 0)), fc=(r, g, b, 0.1), ec=(0, 0, 0, 1), lw=2)
        ax2.add_artist(Rectangle)

    ax1.set_xlim(-2, 35)
    ax1.set_ylim(0, 2)
    ax2.set_xlim(-2, 35)
    ax2.set_ylim(0, 2)
    plt.show()
