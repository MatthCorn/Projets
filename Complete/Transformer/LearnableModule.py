import torch.nn as nn

class LearnableParameters(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = nn.parameter.Parameter(param)

    def forward(self):
        return self.param


if __name__ == '__main__':
    import torch
    from torch.optim import Adam

    class TestNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(10, 3)
            self.param = LearnableParameters(torch.normal(0, 1, (1, 10)))

        def forward(self, x):
            x = x + self.param()
            x = self.lin(x)
            return x


    N = TestNet()
    Input = torch.normal(0, 1, (50, 10))
    opt = Adam(N.parameters())
    Er = torch.norm(N(Input))
    Er.backward()
    opt.step()

