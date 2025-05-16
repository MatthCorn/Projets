import torch
from torch.func import functional_call
from tqdm import tqdm

def forward_with_alphas(model, x, alphas):
    named_params = [(name, p) for name, p in model.named_parameters()]

    patched_params = {
        name: p * a for (name, p), a in zip(named_params, alphas)
    }

    return functional_call(model, patched_params, (x,))

def gradient_variance(loss, params_model, alphas):
    g_params = torch.autograd.grad(loss, params_model, create_graph=True)
    grad = [g_p / a for g_p, a in zip(g_params, alphas)]
    return torch.var(torch.stack([(g**2).sum() for g in grad]))


def novar2_init(model, criterion, x_size, y_size, generator=None, lr=0.1, steps=500, device=torch.device('cpu')):
    alphas_hist = []
    factor_hist = []
    grad_hist = []

    model.eval()
    model.to(device)

    alphas = torch.tensor([1.] * len(list(model.parameters())), requires_grad=True, device=device)

    for i in tqdm(range(steps)):
        if generator is not None:
            input, target = generator(x_size[0], device)
        else:
            input = torch.normal(0, 1, x_size, device=device)
            target = torch.normal(0, 1, y_size, device=device)

        loss = criterion(forward_with_alphas(model, input, alphas), target)
        g_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad = [g_p / a for g_p, a in zip(g_params, alphas)]
        grad_amp = torch.stack([g.norm() for g in grad])
        factor = torch.var(grad_amp) / torch.mean(grad_amp) ** 2
        g = torch.autograd.grad(factor, alphas)

        alphas = torch.maximum(alphas - lr * g[0], 0.01 * torch.ones(alphas.shape, device=device))

        factor_hist.append(factor.item())
        alphas_hist.append(alphas.cpu().tolist())
        grad_hist.append(grad_amp.cpu().tolist())

    print(alphas.data)
    import matplotlib.pyplot as plt
    for i in range(len(alphas)):
        plt.plot([alphas[i] for alphas in alphas_hist])
    plt.show()
    for i in range(len(alphas)):
        plt.plot([grad[i] for grad in grad_hist])
    plt.show()
    plt.plot(factor_hist)
    plt.show()

if __name__ == '__main__':
    from Eusipco.Transformer import Network
    model = Network(n_encoder=3, d_in=10, d_att=64, WidthsEmbedding=[], dropout=0)

    criterion = torch.nn.MSELoss()
    x_size = (1000, 10, 10)
    y_size = (1000, 10, 10)

    from Eusipco.DataMaker import MakeData
    generator = lambda x, d: (vec.to(device=d) for vec in MakeData(NVec=10, DVec=10, NData=x))
    novar2_init(model, criterion, x_size, y_size, generator=generator, lr=0.0001, steps=60000, device=torch.device('cuda'))
