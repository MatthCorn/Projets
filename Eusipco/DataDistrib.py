import torch
import numpy as np

control = True
n_input = 1000
len_seq = 300
d_vec = 15
distrib = 'log'

weight = torch.normal(0, 1, (d_vec,))
weight = weight / torch.norm(weight)

grid_size_abs = 40
grid_size_ord = 40
x_min, x_max = 1e-3, 3
y_min, y_max = -1.5, 1.5

if control:
    if distrib == 'uniform':
        f = lambda x: x
        g = lambda x: x
    elif distrib == 'log':
        f = lambda x: np.log(x)
        g = lambda x: np.exp(x)

    mean = (y_max - y_min) * torch.rand(n_input) + y_min
    std = g((f(x_max) - f(x_min)) * torch.rand(n_input) + f(x_min))

    mean, std = mean.reshape(-1, 1), std.reshape(-1, 1)

    alpha = torch.normal(0, 1, (n_input, len_seq)) * std + mean

    sequences = torch.normal(0, 1, (n_input, len_seq, d_vec)) * alpha.unsqueeze(-1)
    uncontroled_scores = torch.matmul(sequences, weight)
    sequences = sequences + (alpha - uncontroled_scores).unsqueeze(-1) * weight.view(1, 1, d_vec)

else:
    sequences = torch.normal(0, 1, (n_input, len_seq, d_vec,))

scores = torch.matmul(sequences, weight)

mean_scores = torch.mean(scores, dim=-1)
std_scores = torch.std(scores, dim=-1)

x_indices = ((torch.log(std_scores) - np.log(x_min)) / (np.log(x_max) - np.log(x_min)) * grid_size_abs).long()
y_indices = ((mean_scores - y_min) / (y_max - y_min) * grid_size_ord).long()

x_indices = torch.clamp(x_indices, 0, grid_size_abs - 1)
y_indices = torch.clamp(y_indices, 0, grid_size_ord - 1)

grid = torch.zeros((grid_size_ord, grid_size_abs), dtype=torch.int32)
for x, y in zip(x_indices, y_indices):
    grid[y, x] += 1

import matplotlib.pyplot as plt
plt.imshow(grid)

plot_x_ticks = np.exp(torch.linspace(np.log(x_min), np.log(x_max), 7))
x_ticks = torch.linspace(0, grid_size_abs, 7)
plot_y_ticks = torch.linspace(y_min, y_max, 7)
y_ticks = torch.linspace(0, grid_size_ord, 7)

plt.gca().set_xticks(x_ticks)
plt.gca().set_xticklabels([f"{val:.0e}" for val in plot_x_ticks])
plt.gca().set_yticks(y_ticks)
plt.gca().set_yticklabels([f"{val:.1f}" for val in plot_y_ticks])

plt.show()
