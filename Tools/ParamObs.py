import torch
from torch.nn import Linear, Dropout, LayerNorm
from Complete.Transformer.LearnableModule import LearnableParameters

class DictParamObserver(dict):
    def __init__(self, network):
        super().__init__()
        self.NetToDict(network)

    def NetToDict(self, network):
        for name, module in network.named_children():
            if isinstance(module, (Dropout, LayerNorm,)):
                self.__setitem__(name, module._get_name())

            elif isinstance(module, (Linear,)):
                weight = module.weight.data
                try:
                    bias = module.bias.data
                    self.__setitem__(name, {'weight': [float(weight.mean()), float(weight.std())],
                                            'bias': [float(bias.mean()), float(bias.std())]})
                except:
                    self.__setitem__(name, {'weight': [float(weight.mean()), float(weight.std())], 'bias': None})

            elif isinstance(module, (LearnableParameters,)):
                self.__setitem__(name, {'param': [float(module.param.mean()), float(module.param.std())]})

            else:
                self.__setitem__(name, DictParamObserver(module))

def RecInitParam(network, ParamObs):
    if isinstance(network, (Linear,)):
        wmean, wstd = ParamObs['weight']
        weight = torch.normal(wmean, wstd, network.weight.data.shape)
        network.weight.data = weight
        if ParamObs['bias'] is not None:
            bmean, bstd = ParamObs['bias']
            bias = torch.normal(bmean, bstd, network.bias.data.shape)
            network.bias.data = bias

    elif isinstance(network, (LearnableParameters,)):
        pmean, pstd = ParamObs['param']
        param = torch.normal(pmean, pstd, network.param.shape)
        network.param = torch.nn.Parameter(param)

    else:
        for name, module in network.named_children():
            RecInitParam(module, ParamObs[name])

if __name__ == '__main__':
    N = torch.nn.Sequential(torch.nn.Linear(2, 5), torch.nn.Linear(5, 2), LearnableParameters(torch.normal(0, 1, (2, 5))))
    D = DictParamObserver(N)
    print(N[0].weight.data)
    print(N[2].param.data)
    RecInitParam(N, D)
    print(N[0].weight.data)
    print(N[2].param.data)
