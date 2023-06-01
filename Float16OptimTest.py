import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def classe(x):
    return (torch.norm(x, dim=-1) < 1).to(torch.long).to(torch.device('cuda'))

res = torch.nn.Sequential(torch.nn.Linear(2, 10),
                          torch.nn.ReLU(),
                          torch.nn.Linear(10, 10),
                          torch.nn.ReLU(),
                          torch.nn.Linear(10, 2))
res.to(torch.device('cuda'))

optimizer = torch.optim.Adam(res.parameters())
loss = torch.nn.CrossEntropyLoss()

Scaler = True

# avec scaler
if Scaler:
    scaler = torch.cuda.amp.GradScaler()
    err_list = []
    for i in tqdm(range(2000)):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            x = 1.5 * torch.rand(100, 2, device=torch.device('cuda'))
            cl = classe(x)
            out = res(x)
            error = loss(out, cl)
        scaler.scale(error).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        err_list.append(float(error))
    plt.plot(err_list)
    plt.show()

# sans scaler
else:
    err_list = []
    for i in tqdm(range(2000)):
        if i % 100 == 0:
            print(i // 100)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            x = 1.5 * torch.rand(100, 2, device=torch.device('cuda'))
            cl = classe(x)
            error = loss(res(x), cl)
        error.backward()
        optimizer.step()
        err_list.append(float(error))
    plt.plot(err_list)
    plt.show()
