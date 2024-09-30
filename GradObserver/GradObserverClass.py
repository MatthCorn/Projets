import torch

class GradObserver:

    def __init__(self, module, length=10, dist=True):
        self.module = module
        self.dist = dist
        self.mean = 0
        self.std = 0
        self.sample_size = 0
        self.mean_save = []
        self.std_save = []
        self.time = []

        if self.dist:
            self.freq_save = []
            self.bins_save = []
            self.length = length
            self.bins = torch.linspace(self.mean - 2 * self.std, self.mean + 2 * self.std, self.length)
            self.freq = torch.ones(self.length - 1) / (self.length - 1)


    def update(self):
        param = list(self.module.parameters())[0]
        grad = torch.log10(torch.abs(param.grad.detach().cpu()))

        grad[grad == -torch.inf] = 0
        grad[grad == torch.inf] = 0

        self.mean = (self.mean * self.sample_size + grad.mean()) / (self.sample_size + grad.numel())
        self.std = (self.std * (self.sample_size - 1) + grad.std() * (grad.numel() - 1)) / (self.sample_size + grad.numel() - 1)

        if self.dist:
            if self.sample_size == 0:
                Min = torch.quantile(grad, 0.05)
                Max = torch.quantile(grad, 0.95)
                Mean = (Min + Max) / 2
                Diff = (Max - Min) / 2
                # self.bins = torch.linspace(self.mean - 2 * self.std, self.mean + 2 * self.std, self.length)
                self.bins = torch.linspace(Mean - 3 * Diff, Mean + 3 * Diff, self.length)

                # self.bins = torch.linspace(min, max, self.length)

            old_counts = self.freq * self.sample_size
            new_counts = torch.histogram(grad, self.bins)

            current_counts = old_counts + new_counts.hist
            self.freq = current_counts / (self.sample_size + grad.numel())

        self.sample_size = self.sample_size + grad.numel()


    def next(self, j):
        self.time.append(j)
        self.mean_save.append(float(self.mean))
        self.std_save.append(float(self.std))

        self.mean = 0
        self.std = 0
        self.sample_size = 0

        if self.dist:
            self.freq_save.append(self.freq.tolist())
            self.bins_save.append(self.bins.tolist())

            self.bins = torch.arange(self.mean - 2 * self.std, self.mean + 2 * self.std, self.length)
            self.freq = torch.ones(self.length - 1) / (self.length - 1)

    def del_module(self):
        del(self.module)

    def __repr__(self):
        return '{' + 'GradObserver : {}'.format(self.module.__repr__()) + '}'


def GetType(module):
    return str(type(module)).split("'")[1].split(".")[-1]

class DictGradObserver(dict):
    def __init__(self, network, adam_obs=False):
        super().__init__()
        self.NetToDict(network)

    def NetToDict(self, network):
        for name, module in network.named_children():
            if module._get_name() in ['Dropout', 'LayerNorm', ]:
                self.__setitem__(name, module._get_name())

            elif module._get_name() in ['Linear', 'LearnableParameters', ]:
                self.__setitem__(name, GradObserver(module))

            else:
                self.__setitem__(name, DictGradObserver(module))


    def update(self):
        for item in self.values():
            if GetType(item) in ['GradObserver', 'DictGradObserver', ]:
                item.update()

    def next(self, j):
        for item in self.values():
            if GetType(item) in ['GradObserver', 'DictGradObserver', ]:
                item.next(j)

    def del_module(self):
        for item in self.values():
            if GetType(item) in ['GradObserver', 'DictGradObserver', ]:
                item.del_module()