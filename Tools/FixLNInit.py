import torch
from tqdm import tqdm
from math import sqrt
from torch.func import functional_call

def forward_with_alphas(model, x, alphas):
    named_params = [(name, p) for name, p in model.named_parameters()]

    patched_params = {
        name: p * a for (name, p), a in zip(named_params, alphas)
    }

    return functional_call(model, patched_params, (x,))

def eval_method(model, criterion, x_size, y_size, data_generator=None, parameters=None, device=torch.device('cpu')):
    if data_generator is not None:
        input, _ = data_generator(x_size[0], device)
    else:
        input = torch.normal(0, 1, x_size, device=device)

    layernorm_inputs = []
    def save_input_hook(module, input, output):
        layernorm_inputs.append(input[0].detach())

    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            hooks.append(module.register_forward_hook(save_input_hook))

    with torch.no_grad:
        if parameters is not None:
            y = functional_call(model, parameters, (input,))
        else:
            y = model(input)

    loss = 0.
    while len(layernorm_inputs) > 0:
        ln_input = layernorm_inputs.pop()
        loss = loss + (ln_input.norm(dim=-1).mean() - sqrt(ln_input.shape[-1])) ** 2

    for hook in hooks:
        hook.remove()

def fix_ln_init(model, criterion, x_size, y_size, data_generator=None, lr=0.1, steps=500,
              device=torch.device('cpu'), additional_eval=[], freq_eval=50, plot=False):
    alphas_hist = []
    loss_hist = []
    additional_hist = [[] for _ in additional_eval]

    model.eval()
    model.to(device)

    alphas = torch.tensor([1.] * len(list(model.parameters())), requires_grad=True, device=device)

    layernorm_inputs = []
    def save_input_hook(module, input, output):
        layernorm_inputs.append(input[0])

    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            hooks.append(module.register_forward_hook(save_input_hook))

    for i in tqdm(range(steps)):
        loss, alphas = fix_ln_init_step(model, criterion, x_size, y_size, alphas, layernorm_inputs,
                                        data_generator=data_generator, lr=lr, device=device)
        alphas_hist.append(alphas.cpu().tolist())
        loss_hist.append(loss)

        if not i % freq_eval:
            for j in range(len(additional_eval)):
                for hook in hooks:
                    hook.remove()

                named_params = [(name, p) for name, p in model.named_parameters()]

                patched_params = {
                    name: p * a for (name, p), a in zip(named_params, alphas)
                }

                additional_hist[j].append(additional_eval[j](model, criterion, x_size, y_size, parameters=patched_params,
                                                             data_generator=data_generator, device=device))

                hooks = []
                for module in model.modules():
                    if isinstance(module, torch.nn.LayerNorm):
                        hooks.append(module.register_forward_hook(save_input_hook))

    for hook in hooks:
        hook.remove()

    for i, layer in enumerate(model.parameters()):
        layer.data.mul_(alphas[i])

    if plot:
        import matplotlib.pyplot as plt
        for i in range(len(alphas)):
            plt.plot([alphas[i] for alphas in alphas_hist])
        plt.show()
        plt.plot(loss_hist)
        plt.show()
        for hist in additional_hist:
            plt.plot(hist)
            plt.show()

def fix_ln_init_step(model, criterion, x_size, y_size, alphas, layernorm_inputs, data_generator=None, lr=0.1, device=torch.device('cpu')):
        if data_generator is not None:
            input, target = data_generator(x_size[0], device)
        else:
            input = torch.normal(0, 1, x_size, device=device)
            target = torch.normal(0, 1, y_size, device=device)

        y = forward_with_alphas(model, input, alphas)
        loss = 0.
        for ln_input in layernorm_inputs:
            loss = loss + (ln_input.norm(dim=-1).mean() - sqrt(ln_input.shape[-1])) ** 2

        grad = torch.autograd.grad(loss, alphas)
        alphas = torch.maximum(alphas - lr * grad[0], 0.01 * torch.ones(alphas.shape, device=device))

        while layernorm_inputs != []:
            layernorm_inputs.pop()

        return float(loss), alphas


if __name__ == '__main__':
    from Eusipco.Transformer import Network
    model = Network(n_encoder=3, d_in=10, d_att=64, WidthsEmbedding=[], dropout=0)

    criterion = torch.nn.MSELoss()
    x_size = (1000, 10, 10)
    y_size = (1000, 10, 10)

    from Eusipco.DataMaker import MakeData
    data_generator = lambda x, d: (vec.to(device=d) for vec in MakeData(NVec=10, DVec=10, NData=x))
    from Tools.MetaInit import eval_method as eval_metainit
    from Tools.GradInit import eval_method as eval_gradinit
    from Tools.NovarInit import eval_method as eval_novarinit
    fix_ln_init(model, criterion, x_size, y_size, data_generator=None, lr=0.0001, steps=10000,
                device=torch.device('cuda'), additional_eval=[eval_metainit, eval_gradinit, eval_novarinit], plot=True)
