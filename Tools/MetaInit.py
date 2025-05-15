import torch

def gradient_quotient(loss, params, eps=1e-5):
    grad = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)
    prod = torch.autograd.grad(sum([(g**2).sum() / 2 for g in grad]), params, retain_graph=True, create_graph=True)
    out = sum([((g - p) / (g + eps * (2*(g >= 0).float() - 1).detach()) - 1).abs().sum() for g, p in zip(grad, prod)])
    return out / sum([p.data.nelement() for p in params])

def meta_init(model, criterion, x_size, y_size, generator=None, lr=0.1, momentum=0.9, steps=500, eps=1e-5, device=torch.device('cpu')):
    hist_gq = []
    hist_norm = []

    model.eval()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad and len(p.size()) >= 2]
    memory = [0] * len(params)
    for i in range(steps):
        if generator is not None:
            input, target = generator(x_size[0], device)
        else:
            input = torch.normal(0, 1, x_size, device=device)
            target = torch.normal(0, 1, y_size, device=device)
        loss = criterion(model(input), target)
        gq = gradient_quotient(loss, list(model.parameters()), eps)
        grad = torch.autograd.grad(gq, params)

        layerwise_norm = []
        for j, (p, g_all) in enumerate(zip(params, grad)):
            norm = p.data.norm().item()
            g = torch.sign((p.data * g_all).sum() / norm)
            memory[j] = momentum * memory[j] - lr * g.item()
            new_norm = norm + memory[j]
            p.data.mul_(new_norm / norm)

            layerwise_norm.append(new_norm)
        print("%d/GQ = %.2f" % (i, gq.item()))

        hist_gq.append(gq.item())
        hist_norm.append(layerwise_norm)

    import matplotlib.pyplot as plt
    plt.plot(hist_gq)
    plt.show()
    for i in range(len(params)):
        plt.plot([norm[i] for norm in hist_norm])
    plt.show()

if __name__ == '__main__':
    from Eusipco.Transformer import Network
    model = Network(n_encoder=3, d_in=10, d_att=64, WidthsEmbedding=[], dropout=0)

    criterion = torch.nn.MSELoss()
    x_size = (100, 10, 10)
    y_size = (100, 10, 10)

    from Eusipco.DataMaker import MakeData
    generator = lambda x, d: (vec.to(device=d) for vec in MakeData(NVec=10, DVec=10, NData=x))
    meta_init(model, criterion, x_size, y_size, generator=generator, lr=0.0001, steps=10000, device=torch.device('cuda'))