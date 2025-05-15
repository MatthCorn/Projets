import torch

def gradient_variance(loss, params):
    grad = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)
    var = torch.var(torch.stack([(g**2).sum() for g in grad]))
    return var

def novar_init(model, criterion, x_size, y_size, generator=None, lr=0.1, momentum=0.9, steps=500, device=torch.device('cpu')):
    hist_var = []

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
        gv = gradient_variance(loss, list(model.parameters()))
        grad = torch.autograd.grad(gv, params)
        for j, (p, g_all) in enumerate(zip(params, grad)):
            memory[j] = momentum * memory[j] + g_all
            p.data = p.data - lr * memory[j]
        print("%d/GV = %.4f" % (i, gv.item()))
        hist_var.append(gv.item())

    import matplotlib.pyplot as plt
    plt.plot(hist_var)
    plt.show()

if __name__ == '__main__':
    from Eusipco.Transformer import Network
    model = Network(n_encoder=3, d_in=10, d_att=64, WidthsEmbedding=[], dropout=0)

    criterion = torch.nn.MSELoss()
    x_size = (1000, 10, 10)
    y_size = (1000, 10, 10)

    from Eusipco.DataMaker import MakeData
    generator = lambda x, d: (vec.to(device=d) for vec in MakeData(NVec=10, DVec=10, NData=x))
    novar_init(model, criterion, x_size, y_size, generator=None, lr=0.0001, steps=2000, device=torch.device('cuda'))