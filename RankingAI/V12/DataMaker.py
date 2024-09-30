import torch

def GetSelected(Input, WeightsF, WeightsN, Threshold=0.14):
    if WeightsF is None:
        WeightsF = torch.tensor([1.]*Input.shape[-1])
    if WeightsN is None:
        WeightsN = torch.tensor([1.] * Input.shape[-1])
    ValuesF = torch.matmul(Input, WeightsF)
    DistF = torch.abs(ValuesF.unsqueeze(-1) - ValuesF.unsqueeze(-2)) < Threshold
    ValuesN = torch.matmul(Input, WeightsN)
    SupN = (ValuesN.unsqueeze(-1) > ValuesN.unsqueeze(-2))
    InterF = torch.sum(DistF * SupN, dim=-2)
    Mask = InterF > 0

    return Mask


def MakeData(NInput=10, DVec=10, sigma=1, NData=1000, WeightsF=None, WeightsN=None, Threshold=0.14):
    Input = torch.normal(torch.zeros(NData, NInput, DVec), sigma*torch.ones(NData, NInput, DVec))

    Mask = GetSelected(Input, WeightsF, WeightsN, Threshold=Threshold)
    Output = Mask.to(torch.float)
    return Input, Output.unsqueeze(-1)


if __name__ == '__main__':
    MakeData(10, 2, 1, 1, WeightsF=torch.tensor([1., 0.]), WeightsN=torch.tensor([0., 1.]), Threshold=0.14)

    if False:
        # Trouver le bon seuil
        Threshold = 10
        while True:
            I, O = MakeData(10, 2, 1, 10000, WeightsF=torch.tensor([1., 0.]), WeightsN=torch.tensor([0., 1.]), Threshold=Threshold)
            mean = torch.mean(torch.sum(O, dim=-1))
            if mean < 5:
                break
            Threshold *= 0.99
        print(Threshold)

