import torch
from torch.func import functional_call
from tqdm import tqdm

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

def eval_method(model, criterion, x_size, y_size, data_generator=None, lr=0.1, gamma=0.1, parameters=None, device=torch.device('cpu')):
    if data_generator is not None:
        first_set = tuple(data_generator(x_size[0], device))
        second_set = tuple(data_generator(x_size[0], device))
    else:
        first_set = (torch.normal(0, 1, x_size, device=device), torch.normal(0, 1, y_size, device=device))
        second_set = (torch.normal(0, 1, x_size, device=device), torch.normal(0, 1, y_size, device=device))
    if parameters is not None:
        first_loss = criterion(functional_call(model, parameters, (first_set[0],)), first_set[1])
        grad = torch.autograd.grad(first_loss, list(parameters.values()))
        norm_grad = sum([g.norm().index() for g in grad])
        if norm_grad > gamma:
            grad = grad * gamma / norm_grad

        patched_params = {
            name: p - lr * g for (name, p), g in zip(parameters.items(), grad)
        }

        second_loss = criterion(functional_call(model, patched_params, (second_set[0],)), second_set[1]).index()
        return first_loss.index() - second_loss
    else:
        first_loss = criterion(model(first_set[0]), first_set[1])
        grad = torch.autograd.grad(first_loss, model.parameters())
        norm_grad = sum([float(g.norm()) for g in grad])
        if norm_grad > gamma:
            grad = [g * gamma / norm_grad for g in grad]

        named_params = [(name, p) for name, p in model.named_parameters()]
        patched_params = {
            name: p - lr * g for (name, p), g in zip(named_params, grad)
        }
        second_loss = criterion(functional_call(model, patched_params, (second_set[0],)), second_set[1])
        return float(first_loss - second_loss)



def scaled_grad(loss, params_model, alphas):
    g_params = torch.autograd.grad(loss, params_model, create_graph=True)
    return [g_p / a for g_p, a in zip(g_params, alphas)]

def grad_init(model, criterion, x_size, y_size, data_generator=None, gamma=0.1, lr=0.1, steps=500,
              device=torch.device('cpu'), additional_eval=[], freq_eval=50):
    alphas_hist = []
    loss_hist = []
    additional_hist = [[] for _ in additional_eval]

    model.eval()
    model.to(device)

    alphas = torch.tensor([1.] * len(list(model.parameters())), requires_grad=True, device=device)

    for i in tqdm(range(steps)):
        alphas, loss = grad_init_step(model, criterion, x_size, y_size, alphas, data_generator=data_generator,
                                      gamma=gamma, lr=lr, device=device)
        alphas_hist.append(alphas.cpu().tolist())
        loss_hist.append(loss)

        if not i % freq_eval:
            for j in range(len(additional_eval)):
                additional_hist[j].append(additional_eval[j](model, criterion, x_size, y_size, data_generator=data_generator, device=device))

    import matplotlib.pyplot as plt
    for i in range(len(alphas)):
        plt.plot([alphas[i] for alphas in alphas_hist])
    plt.show()
    plt.plot(loss_hist)
    plt.show()
    for hist in additional_hist:
        plt.plot(hist)
        plt.show()

def grad_init_step(model, criterion, x_size, y_size, alphas, data_generator=None, gamma=0.1, lr=0.1, device=torch.device('cpu')):
        if data_generator is not None:
            input, target = data_generator(x_size[0], device)
        else:
            input = torch.normal(0, 1, x_size, device=device)
            target = torch.normal(0, 1, y_size, device=device)
        loss = criterion(forward_with_alphas(model, input, alphas), target)
        sg = scaled_grad(loss, list(model.parameters()), alphas)
        nsg = sum([g.norm() for g in sg])
        if nsg > gamma:
            g = torch.autograd.grad(nsg, alphas)
            alphas = alphas - lr * g[0]
        else:
            if data_generator is not None:
                input, target = data_generator(x_size[0], device)
            else:
                input = torch.normal(0, 1, x_size, device=device)
                target = torch.normal(0, 1, y_size, device=device)
            step = [-lr * g for g in sg]
            loss = criterion(next_forward(model, input, alphas, step), target)
            g = torch.autograd.grad(loss, alphas)
            alphas = alphas - lr * g[0]

        alphas = torch.maximum(alphas, 0.01 * torch.ones(alphas.shape, device=device))

        return alphas, loss.item()
        alphas_hist.append(alphas.cpu().tolist())


if __name__ == '__main__':
    from Eusipco.Transformer import Network
    model = Network(n_encoder=3, d_in=10, d_att=64, WidthsEmbedding=[], dropout=0)

    criterion = torch.nn.MSELoss()
    x_size = (1000, 10, 10)
    y_size = (1000, 10, 10)

    from Eusipco.DataMaker import MakeData
    data_generator = lambda x, d: (vec.to(device=d) for vec in MakeData(NVec=10, DVec=10, NData=x))

    grad_init(model, criterion, x_size, y_size, data_generator=None, gamma=5, lr=0.0001, steps=5000, device=torch.device('cuda'))
