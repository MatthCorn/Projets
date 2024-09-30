from StepByStep.NoiseLevel.Network import Network
import torch
from tqdm import tqdm

torch.cuda.set_device(1)

size_data = 20000
batch_size = 1000
dim = 10
seq_len = 5
vec_dir = torch.rand(size_data, dim, 1)
mul = 1000 * torch.rand(size_data, seq_len, 1)

data = torch.matmul(vec_dir, mul.transpose(1, 2)).transpose(1, 2)
# data.shape = (batch_size, seq_len, dim)

noise = torch.rand(size_data, 1, 1)


noise_add = torch.normal(torch.zeros(size_data, seq_len, dim))
noise_add = noise * noise_add
data = data + noise_add

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Net = Network(n_encoder=7, max_len=seq_len, d_in=dim, d_att=64, n_heads=4, PE=False, norm='n').to(device)
data = data.to(device)
noise = noise.to(device)

opt = torch.optim.Adam(Net.parameters())

err_list = []
n_epoch = 50

n_batch = int(size_data/batch_size)


for _ in tqdm(range(n_epoch)):
    for i in range(n_batch):
        opt.zero_grad(set_to_none=True)

        pred = Net(data[i*batch_size:(i+1)*batch_size])
        err = torch.norm(pred-noise[i*batch_size:(i+1)*batch_size])
        err.backward()
        opt.step()

        err_list.append(float(err)/batch_size)


import matplotlib.pyplot as plt
plt.plot(err_list)
plt.plot([float(torch.std(noise))]*len(err_list), 'black')
plt.ylim(bottom=0)
plt.show()

