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
        self.D = [] # contient un encodage des vecteurs sélectionnés à chaque itération par le simulateur

        self.sensor_simulator = SensorSimulator(dim=dim, sensitivity=sensitivity)
        if seed is not None:
            np.random.seed(seed)

    def Step(self):
        v = np.random.normal(0, 1, self.dim).tolist()

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

            Selected = self.sensor_simulator.V.numpy()
            Match = np.array([[0, -1]] * self.n)
            if len(Selected) != 0:
                History = np.array([x[:self.dim] for x in self.L])
                Dist = np.linalg.norm(np.expand_dims(Selected, 1) - np.expand_dims(History, 0), axis=-1)
                Match = Dist.argmin(axis=1) - self.T + 1
                Match = np.pad(np.expand_dims(Match, axis=1), ((0, 0), (0, 1)), 'constant')
            if len(Match) != self.n:
                Match = np.concatenate((Match, np.array([[0, -1]] * (self.n - len(Match)))))
            self.D.append(Match.tolist())
