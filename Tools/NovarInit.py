import torch
from tqdm import tqdm
from torch.func import functional_call
from math import sqrt

def eval_method(model, criterion, x_size, y_size, data_generator=None, parameters=None, device=torch.device('cpu')):
    if data_generator is not None:
        input, target = data_generator(x_size[0], device)
    else:
        input = torch.normal(0, 1, x_size, device=device)
        target = torch.normal(0, 1, y_size, device=device)
    if parameters is not None:
        loss = criterion(functional_call(model, parameters, (input,)), target)
        grad = torch.autograd.grad(loss, list(parameters.values()))
    else:
        loss = criterion(model(input), target)
        grad = torch.autograd.grad(loss, model.parameters())
    grad_amp = torch.stack([g.norm()/sqrt(g.numel()) for g in grad])
    factor = torch.var(grad_amp) / torch.mean(grad_amp) ** 2
    return float(factor)

def novar_init_step(model, criterion, params, memory, x_size, y_size, data_generator=None, lr=0.1, momentum=0.9, device=torch.device('cpu')):
    if data_generator is not None:
        input, target = data_generator(x_size[0], device)
    else:
        input = torch.normal(0, 1, x_size, device=device)
        target = torch.normal(0, 1, y_size, device=device)
    loss = criterion(model(input), target)
    grad = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)
    grad_amp = torch.stack([g.norm() / sqrt(g.numel()) for g in grad])
    factor = torch.var(grad_amp) / torch.mean(grad_amp) ** 2
    grad = torch.autograd.grad(factor, params)
    for j, (p, g_all) in enumerate(zip(params, grad)):
        memory[j] = momentum * memory[j] + g_all
        p.data = p.data - lr * memory[j]
    return factor.item(), grad_amp.cpu().tolist(), [p.data.norm().item() for p in model.parameters()]

def redraw_param(model, p=0.1):
    for layer in model.parameters():
        mu = layer.data.mean()
        sigma = layer.data.std() + 1e-3
        dist = torch.distributions.Normal(mu, sigma)
        mask = torch.bernoulli(p * torch.ones(layer.data.size())).to(layer.device)
        layer.data = (1 - mask) * layer.data + mask * dist.sample((layer.data.size())).to(layer.device)

def novar_init(model, criterion, x_size, y_size, data_generator=None, lr=0.1, momentum=0.9, steps=500,
               device=torch.device('cpu'), additional_eval=[], freq_eval=50, freq_redraw=10, plot=False):
    hist_var = []
    hist_grad = []
    hist_norm_param = []

    additional_hist = [[] for _ in additional_eval]

    model.eval()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad and len(p.size()) >= 2]
    memory = [0] * len(params)
    for i in tqdm(range(steps)):

        factor, grad_amp, norm_param = novar_init_step(model, criterion, params, memory, x_size, y_size,
                                                       data_generator=data_generator, lr=lr, momentum=momentum, device=device)

        hist_var.append(factor)
        hist_grad.append(grad_amp)
        hist_norm_param.append(norm_param)

        if not i % freq_eval:
            for j in range(len(additional_eval)):
                additional_hist[j].append(additional_eval[j](model, criterion, x_size, y_size, data_generator=data_generator, device=device))

        if not i % freq_redraw:
            redraw_param(model, p=0.05)

    if plot:
        import matplotlib.pyplot as plt
        for i in range(len(params)):
            plt.plot([grad[i] for grad in hist_grad])
        plt.show()
        plt.plot(hist_var)
        plt.show()
        for i in range(len(params)):
            plt.plot([norm[i] for norm in hist_norm_param])
        plt.show()
        for hist in additional_hist:
            plt.plot(hist)
            plt.show()

if __name__ == '__main__':
    from Eusipco.Transformer import Network
    model = Network(n_encoder=3, d_in=10, d_att=64, WidthsEmbedding=[], dropout=0)

    criterion = torch.nn.MSELoss()
    x_size = (1000, 10, 10)
    y_size = (1000, 10, 10)

    from Eusipco.DataMaker import MakeData
    from Tools.MetaInit import eval_method as eval_metainit
    from Tools.GradInit import eval_method as eval_gradinit
    data_generator = lambda x, d: (vec.to(device=d) for vec in MakeData(NVec=10, DVec=10, NData=x))
    novar_init(model, criterion, x_size, y_size, data_generator=None, lr=0.001, steps=20000,
               device=torch.device('cuda'), additional_eval=[eval_gradinit, eval_metainit], plot=True)