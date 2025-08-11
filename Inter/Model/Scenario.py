import numpy as np
from Inter.Model.Sensor import Simulator as SensorSimulator
from functools import partial

class Simulator:
    def __init__(self, n, N, dim, n_sat=5, n_mes=6, sensitivity=0.2, seed=None, WeightF=None, WeightL=None, model_path=None):
        self.n = n  # Nombre de vecteurs présents simultanément
        self.N = N  # Nombre total de vecteurs dans le scénario
        self.n_sat = n_sat # Nombre max de vecteurs pouvant être détectés simultanément
        self.n_mes = n_mes # Nombre max de mesureurs pouvant existés simultanément
        self.dim = dim
        self.T = 0
        self.V = [] # contient tous les vecteurs présents simultanément à un instant donné
        self.A = [] # contient l'âge de chaque vecteur présent à un instant donné
        self.L = [] # contient chaque vecteur du scénario ainsi que leurs ages à leurs disparitions
        self.P = [] # contient un encodage des vecteurs sélectionnables à chaque itération par le simulateur
        self.D = [] # contient un encodage des vecteurs sélectionnés à chaque itération par le simulateur

        self.weight_f = WeightF if WeightF is not None else np.array([1., 0.] + [0.] * (self.dim - 2))
        self.weight_f = self.weight_f / np.linalg.norm(self.weight_f)
        self.weight_l = WeightL if WeightL is not None else np.array([0., 1.] + [0.] * (self.dim - 2))
        self.weight_l = self.weight_l / np.linalg.norm(self.weight_l)
        self.gamma = np.matmul(self.weight_f, self.weight_l)
        self.sensor_simulator = SensorSimulator(dim=dim, sensitivity=sensitivity, n_sat=n_sat, n_mes=n_mes, WeightF=self.weight_f, WeightL=self.weight_l)
        if seed is not None:
            np.random.seed(seed)
        if model_path is not None:
            self.sensor_simulator.load_model(model_path)

    def step(self):
        res = 0
        if (self.T == 0) or (len(self.V) > 0):
            add = True

            if self.T >= self.n:
                # sélection d'un indice dans les vecteurs du palier, correspondant au vecteur disparaissant
                m = len(self.V)
                p = [(1 + a) / (m + sum(np.array(self.A))) for a in self.A]
                k = np.random.choice(range(m), p=p)

                # on supprime le vecteur disparu de self.V, et on retient son âge depuis self.A
                a_out, v_out = self.A.pop(k), self.V.pop(k)

                # on inscrit l'âge du vecteur juste disparu dans son instance présente dans self.L
                i = self.L.index(v_out)
                self.L[i] += [a_out]

            if self.T < self.N:
                res = 1
                v = self.make_vector()
                # on ajoute le nouveau vecteur dans self.L et self.V, ainsi que son âge actuel dans self.A
                self.L.append(v)
                self.V.append(v)
                self.A.append(0)

            self.T += 1
            for i in range(len(self.A)):
                self.A[i] += 1

            self.add(np.array(self.V), self.P)
        else:
            add = False
        self.sensor_simulator.Process(self.V)
        if add:
            self.add(self.sensor_simulator.V.numpy(), self.D)
        return res

    def run(self):
        while self.sensor_simulator.running:
            self.step()

    def make_vector(self):
        return np.random.normal(0, 1, self.dim).tolist()


    def add(self, Selected, location):
        n = self.n if location is self.P else self.n_sat
        Match = np.array([[0, -1]] * n)
        if len(Selected) != 0:
            History = np.array([x[:self.dim] for x in self.L])
            Dist = np.linalg.norm(np.expand_dims(Selected, 1) - np.expand_dims(History, 0), axis=-1)
            Match = Dist.argmin(axis=1) - self.T + 1
            Match = np.pad(np.expand_dims(Match, axis=1), ((0, 0), (0, 1)), 'constant')
        if len(Match) != n:
            Match = np.concatenate((Match, np.array([[0, -1]] * (n - len(Match)))))
        location.append(Match.tolist())

class BiasedSimulatorTemplate():
    def __init__(self, std, mean, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.std = std
        self.mean = mean

        self.mean_f = None
        self.std_f = None
        self.mean_l = None
        self.std_l = None

        self.alpha = None
        self.beta = None
        self.constrain_sequence()

    def constrain_sequence(self):
        lbd_mean = np.random.rand()
        lbd_std = np.random.rand() * (1 - abs(self.gamma)) / (1 + abs(self.gamma)) + (abs(self.gamma) / (1 + abs(self.gamma)))
        eps_mean = np.random.randint(0, 2)

        std_F = self.std * lbd_std
        std_N = self.std * (1 - lbd_std)
        mean_F = (-1) ** (eps_mean + (self.mean > 0)) * self.mean * lbd_mean
        mean_N = (-1) ** (eps_mean + 2 * (self.mean > 0)) * self.mean * (1 - lbd_mean)

        self.alpha = partial(np.random.normal,
                             (mean_F - self.gamma * mean_N) / (1 - self.gamma ** 2),
                             np.sqrt((std_F ** 2 - self.gamma ** 2 * std_N ** 2) / (1 - self.gamma ** 4)))

        self.beta = partial(np.random.normal,
                            (mean_N - self.gamma * mean_F) / (1 - self.gamma ** 2),
                            np.sqrt((std_N ** 2 - self.gamma ** 2 * std_F ** 2) / (1 - self.gamma ** 4)))

    def make_vector(self):
        alpha = self.alpha()
        beta = self.beta()

        raw_vector = np.random.randn(self.dim) * np.sqrt(alpha ** 2 + beta ** 2)

        values_f = np.matmul(raw_vector, self.weight_f)
        values_l = np.matmul(raw_vector, self.weight_l)

        ortho_vector = (
                raw_vector
                - (values_l - self.gamma * values_f) / (1 - self.gamma ** 2) * self.weight_l
                - (values_f - self.gamma * values_l) / (1 - self.gamma ** 2) * self.weight_f
        )

        vector = ortho_vector + alpha * self.weight_f + beta * self.weight_l

        return vector.tolist()

class FreqBiasedSimulatorTemplate():
    def __init__(self, std, mean, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.std = std
        self.mean = mean

    def make_vector(self):
        f = np.random.randn() * self.std + self.mean

        raw_vector = np.random.randn(self.dim) * abs(f)

        raw_f = np.matmul(raw_vector, self.weight_f)

        vector = raw_vector + (f - raw_f) * self.weight_f

        return vector.tolist()

class BiasedSimulator(BiasedSimulatorTemplate, Simulator):
    pass

class FreqBiasedSimulator(FreqBiasedSimulatorTemplate, Simulator):
    pass