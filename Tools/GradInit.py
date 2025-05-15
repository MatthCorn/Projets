import torch
from torch.func import functional_call

def forward_with_alphas(model, x, alphas):
    named_params = [(name, p) for name, p in model.named_parameters()]

    patched_params = {
        name: p * a for (name, p), a in zip(named_params, alphas)
    }

    return functional_call(model, patched_params, (x,))

def next_forward(model, x, alphas, step):
    named_params = [(name, p) for name, p in model.named_parameters()]

    patched_params = {
        name: p * a - s for (name, p), a, s in zip(named_params, alphas, step)
    }

    return functional_call(model, patched_params, (x,))

def scaled_grad(loss, params_model, alphas):
    g_params = torch.autograd.grad(loss, params_model, create_graph=True)
    return [g_p / a for g_p, a in zip(g_params, alphas)]

def grad_init(model, criterion, x_size, y_size, generator=None, gamma=0.1, lr=0.1, steps=500, device=torch.device('cpu')):
    alphas_hist = []
    loss_hist = []

    model.eval()
    model.to(device)

    alphas = torch.tensor([1.] * len(list(model.parameters())), requires_grad=True, device=device)

    for i in range(steps):
        if generator is not None:
            input, target = generator(x_size[0], device)
        else:
            input = torch.normal(0, 1, x_size, device=device)
            target = torch.normal(0, 1, y_size, device=device)
        loss = criterion(forward_with_alphas(model, input, alphas), target)
        sg = scaled_grad(loss, list(model.parameters()), alphas)
        nsg = sum([g.norm() for g in sg])
        if nsg > gamma:
            g = torch.autograd.grad(nsg, alphas)
            alphas = alphas - lr * g[0]
            print("%d/nsg = %.2f" % (i, nsg.item()))
        else:
            if generator is not None:
                input, target = generator(x_size[0], device)
            else:
                input = torch.normal(0, 1, x_size, device=device)
                target = torch.normal(0, 1, y_size, device=device)
            step = [-lr * g for g in sg]
            loss = criterion(next_forward(model, input, alphas, step), target)
            g = torch.autograd.grad(loss, alphas)
            alphas = torch.maximum(alphas - lr * g[0], 0.01 * torch.ones(alphas.shape, device=device))

            print("%d/loss = %.2f" % (i, loss.item()))

            loss_hist.append(loss.item())
        alphas_hist.append(alphas.cpu().tolist())

    print(alphas.data)
    import matplotlib.pyplot as plt
    for i in range(len(alphas)):
        plt.plot([alphas[i] for alphas in alphas_hist])
    plt.show()
    plt.plot(loss_hist)
    plt.show()

if __name__ == '__main__':
    from Eusipco.Transformer import Network
    model = Network(n_encoder=3, d_in=10, d_att=64, WidthsEmbedding=[], dropout=0)

    criterion = torch.nn.MSELoss()
    x_size = (1000, 10, 10)
    y_size = (1000, 10, 10)

    from Eusipco.DataMaker import MakeData
    generator = lambda x, d: (vec.to(device=d) for vec in MakeData(NVec=10, DVec=10, NData=x))
    grad_init(model, criterion, x_size, y_size, generator=None, gamma=5, lr=0.0001, steps=20000, device=torch.device('cuda'))
