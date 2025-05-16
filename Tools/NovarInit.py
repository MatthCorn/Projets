import torch
from tqdm import tqdm

def novar_init(model, criterion, x_size, y_size, generator=None, lr=0.1, momentum=0.9, steps=500, device=torch.device('cpu')):
    hist_var = []
    hist_grad = []
    hist_norm_param = []

    model.eval()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad and len(p.size()) >= 2]
    memory = [0] * len(params)
    for i in tqdm(range(steps)):
        if generator is not None:
            input, target = generator(x_size[0], device)
        else:
            input = torch.normal(0, 1, x_size, device=device)
            target = torch.normal(0, 1, y_size, device=device)
        loss = criterion(model(input), target)
        grad = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)
        grad_amp = torch.stack([g.norm() for g in grad])
        factor = torch.var(grad_amp) / torch.mean(grad_amp)**2
        grad = torch.autograd.grad(factor, params)
        for j, (p, g_all) in enumerate(zip(params, grad)):
            memory[j] = momentum * memory[j] + g_all
            p.data = p.data - lr * memory[j]
        hist_var.append(factor.item())
        hist_grad.append(grad_amp.cpu().tolist())
        hist_norm_param.append([p.data.norm().item() for p in model.parameters()])

    import matplotlib.pyplot as plt
    for i in range(len(params)):
        plt.plot([grad[i] for grad in hist_grad])
    plt.show()
    plt.plot(hist_var)
    plt.show()
    for i in range(len(params)):
        plt.plot([norm[i] for norm in hist_norm_param])
    plt.show()

if __name__ == '__main__':
    from Eusipco.Transformer import Network
    model = Network(n_encoder=3, d_in=10, d_att=64, WidthsEmbedding=[], dropout=0)

    criterion = torch.nn.MSELoss()
    x_size = (1000, 10, 10)
    y_size = (1000, 10, 10)

    from Eusipco.DataMaker import MakeData
    generator = lambda x, d: (vec.to(device=d) for vec in MakeData(NVec=10, DVec=10, NData=x))
    novar_init(model, criterion, x_size, y_size, generator=None, lr=0.0001, steps=10000, device=torch.device('cuda'))