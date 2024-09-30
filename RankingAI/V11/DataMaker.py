import torch

def GetSelected(Input, WeightsF, Threshold=0.2):
    if WeightsF is None:
        WeightsF = torch.tensor([1.]*Input.shape[-1])
    Values = torch.matmul(Input, WeightsF)
    Dist = torch.abs(Values.unsqueeze(-1) - Values.unsqueeze(-2))
    Inter = torch.sum(Dist < Threshold, dim=-2)
    Mask = Inter > 1

    return Mask


def MakeData(NInput=10, DVec=10, sigma=1, NData=1000, WeightsF=None, Threshold=0.14):
    Input = torch.normal(torch.zeros(NData, NInput, DVec), sigma*torch.ones(NData, NInput, DVec))

    Mask = GetSelected(Input, WeightsF, Threshold=Threshold)
    Output = Mask.to(torch.float)
    return Input, Output.unsqueeze(-1)


if __name__ == '__main__':
    MakeData(10, 2, 1, 5, WeightsF=torch.tensor([1., 0.]), Threshold=0.14)

    if False:
        # Trouver le bon seuil
        Threshold = 10
        while True:
            I, O = MakeData(10, 2, 1, 10000, WeightsF=torch.tensor([1., 0.]), Threshold=Threshold)
            mean = torch.mean(torch.sum(O, dim=-1))
            if mean < 5:
                break
            Threshold *= 0.99
        print(Threshold)

