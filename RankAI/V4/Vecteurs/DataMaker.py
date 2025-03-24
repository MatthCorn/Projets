import torch
import math

def GetSelected(Input, WeightF, WeightN, Threshold=0.14):
    if WeightF is None:
        WeightF = torch.tensor([1.]*Input.shape[-1])
    if WeightN is None:
        WeightN = torch.tensor([1.] * Input.shape[-1])
    ValuesF = torch.matmul(Input, WeightF)
    DistF = torch.abs(ValuesF.unsqueeze(-1) - ValuesF.unsqueeze(-2)) < Threshold
    ValuesN = torch.matmul(Input, WeightN)
    SupN = (ValuesN.unsqueeze(-1) > ValuesN.unsqueeze(-2))
    InterF = torch.sum(DistF * SupN, dim=-2)
    Mask = InterF > 0

    return Mask

def GetSorted(Input, Mask, WeightN):
    if WeightN is None:
        WeightN = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightN)
    Values[Mask] = -float('inf')
    Orders = Values.argsort(dim=-1, descending=True)
    Output = Input.clone()
    Output[Mask] = 0

    return Output[torch.arange(Output.size(0)).unsqueeze(1), Orders]


def MakeData(NInput=10, DVec=10, mean=0, std=1, NData=1000, WeightF=None, WeightN=None, Threshold=0.14, NOutput=5):
    Input = torch.normal(mean, std, (NData, NInput, DVec))

    Mask = GetSelected(Input, WeightF, WeightN, Threshold=Threshold)
    Output = GetSorted(Input, Mask, WeightN)
    return Input, Output[:, :NOutput]

def MakeTargetedData(NInput=10, DVec=10, NData=1000, WeightF=None, WeightN=None, Threshold=0.14, NOutput=5,
                     mean_min=-10, mean_max=10, std_min=1, std_max=5, distrib='log', plot=False):
    if WeightF is None:
        WeightF = torch.tensor([1.] * DVec)
    if WeightN is None:
        WeightN = torch.tensor([1.] * DVec)
    WeightF = WeightF / torch.norm(WeightF)
    WeightN = WeightN / torch.norm(WeightN)

    gamma = torch.matmul(WeightF, WeightN)

    if plot:
        spacing = lambda x: torch.linspace(0, 1, x)
    else:
        spacing = lambda x: torch.rand(x)

    if distrib == 'uniform':
        f = lambda x: x
        g = lambda x: x
    elif distrib == 'log':
        f = lambda x: math.log(x)
        g = lambda x: torch.exp(x)
    mean = (mean_max - mean_min) * spacing(NData) + mean_min
    std = g((f(std_max) - f(std_min)) * spacing(NData) + f(std_min))
    lbd_mean = torch.rand(NData)
    lbd_std = torch.rand(NData) * (1 - abs(gamma)) / (1 + abs(gamma)) + (abs(gamma) / (1 + abs(gamma)))
    eps_mean = torch.randint(0, 2, (NData,))

    std_F = std * lbd_std
    std_N = std * (1 - lbd_std)
    mean_F = (-1) ** (eps_mean + (mean > 0)) * mean * lbd_mean
    mean_N = (-1) ** (eps_mean + 2 * (mean > 0)) * mean * (1 - lbd_mean)

    if plot:
        mean_N, std_N = torch.meshgrid(mean_N, std_N)
        mean_F, std_F = torch.meshgrid(mean_F, std_F)

        NData = NData ** 2

    mean_N, std_N = mean_N.reshape(-1, 1), std_N.reshape(-1, 1)
    mean_F, std_F = mean_F.reshape(-1, 1), std_F.reshape(-1, 1)

    alpha = (
            torch.normal(0, 1, (NData, NInput))
            * torch.sqrt((std_F ** 2 - gamma ** 2 * std_N ** 2) / (1 - gamma ** 4))
            + ((mean_F - gamma * mean_N) / (1 - gamma ** 2))
    )

    beta = (
            torch.normal(0, 1, (NData, NInput))
            * torch.sqrt((std_N ** 2 - gamma ** 2 * std_F ** 2) / (1 - gamma ** 4))
            + ((mean_N - gamma * mean_F) / (1 - gamma ** 2))
    )

    Raw_Input = torch.normal(0, 1, (NData, NInput, DVec)) * torch.stack((alpha, beta), dim=-1).norm(dim=-1, keepdim=True)

    ValuesF = torch.matmul(Raw_Input, WeightF)
    ValuesN = torch.matmul(Raw_Input, WeightN)

    Ortho_Input = (
            Raw_Input
            - (ValuesN - gamma * ValuesF).unsqueeze(-1) / (1 - gamma ** 2) * WeightN.view(1, 1, DVec)
            - (ValuesF - gamma * ValuesN).unsqueeze(-1) / (1 - gamma ** 2) * WeightF.view(1, 1, DVec)
    )

    Input = Ortho_Input + alpha.unsqueeze(-1) * WeightF.view(1, 1, DVec) + beta.unsqueeze(-1) * WeightN.view(1, 1, DVec)

    Mask = GetSelected(Input, WeightF, WeightN, Threshold=Threshold)
    Output = GetSorted(Input, Mask, WeightN)
    return Input, Output[:, :NOutput], torch.stack((alpha, beta), dim=-1).norm(dim=[-2, -1], keepdim=True)

if __name__ == '__main__':
    DVec = 10
    WeightN = 2 * torch.rand(DVec) - 1
    WeightF = 2 * torch.rand(DVec) - 1

    MakeTargetedData(DVec=DVec, WeightF=WeightF, WeightN=WeightN)