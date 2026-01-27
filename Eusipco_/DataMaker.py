import torch
import math

def GetSorted(Input, Weight):
    if Weight is None:
        Weight = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, Weight.to(Input.device))
    Orders = Values.argsort(dim=-1)
    return Input[torch.arange(Input.size(0)).unsqueeze(1), Orders]

def MakeData(NVec=5, DVec=10, mean=0, std=1, NData=1000, Weight=None):
    Input = torch.normal(mean, std, (NData, NVec, DVec))

    Output = GetSorted(Input, Weight)
    return Input, Output

def MakeTargetedData(NVec=5, DVec=10, mean_min=1e-1, mean_max=1e1, std_min=1e0, std_max=1e1, distrib='log', plot=False, NData=1000, Weight=None, device=torch.device('cpu')):
    if Weight is None:
        Weight = torch.tensor([1.] * DVec)
    Weight = Weight / torch.norm(Weight)
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

    if plot:
        mean, std = torch.meshgrid(mean, std)
        NData = NData ** 2

    mean, std = mean.reshape(-1, 1), std.reshape(-1, 1)

    alpha = torch.normal(0, 1, (NData, NVec)) * std + mean

    Input = torch.normal(0, 1, (NData, NVec, DVec)) * alpha.unsqueeze(-1)

    # centered = Input - Input.mean(dim=-2, keepdim=True)
    # dist = torch.sqrt(torch.sum(centered ** 2, dim=-1) / 10)
    # err = torch.mean(dist, dim=-1, keepdim=True)
    # l1 = err / abs(alpha).mean(dim=-1, keepdim=True)
    #
    # centered = Input - Input.mean(dim=[0, -2], keepdim=True)
    # dist = torch.sqrt(torch.sum(centered ** 2, dim=-1) / 10)
    # err = torch.mean(dist, dim=-1, keepdim=True)
    # l2 = err / torch.std(Input)
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(l1.reshape(50, 50))
    # plt.show()
    # plt.imshow(l2.reshape(50, 50))
    # plt.show()

    uncontroled_values = torch.matmul(Input, Weight.to(Input.device))
    Input = Input + (alpha - uncontroled_values).unsqueeze(-1) * Weight.view(1, 1, DVec)

    Output = GetSorted(Input, Weight)
    return Input.to(device), Output.to(device), abs(alpha).mean(dim=-1, keepdim=True).unsqueeze(-1).to(device)
