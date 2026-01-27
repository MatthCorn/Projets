import torch
import numpy as np

torch.set_default_dtype(torch.float32)

control = True
n_input = 1000000
len_seq = 30
d_vec = 15
distrib = 'log'
plot = 'log'

weight = torch.normal(0, 1, (d_vec,))
weight = weight / torch.norm(weight)

grid_size_abs = 200
grid_size_ord = 200
x_min, x_max = 1e-6, 50
y_min, y_max = -100, 100

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
    sequences = torch.normal(0, 1, (n_input, len_seq, d_vec,)) * x_max / 1.5

scores = torch.matmul(sequences, weight)

mean_scores = torch.mean(scores, dim=-1)
std_scores = torch.std(scores, dim=-1)

if plot == 'log':
    f = np.log
    g = np.exp
elif plot == 'uniform':
    f = lambda x: x
    g = lambda x: x

y_min_plot, y_max_plot = y_min - 10, y_max + 10
x_max_plot, x_min_plot = x_max + 1, x_min

x_indices = ((f(std_scores) - f(x_min_plot)) / (f(x_max_plot) - f(x_min_plot)) * grid_size_abs).long()
y_indices = ((mean_scores - y_min_plot) / (y_max_plot - y_min_plot) * grid_size_ord).long()

x_indices = torch.clamp(x_indices, 0, grid_size_abs - 1)
y_indices = torch.clamp(y_indices, 0, grid_size_ord - 1)

grid = torch.zeros((grid_size_ord, grid_size_abs), dtype=torch.int32)
for x, y in zip(x_indices, y_indices):
    grid[y, x] += 1

import matplotlib.pyplot as plt
plt.grid()
plt.axis('square')
plt.imshow(grid)

plot_x_ticks = g(torch.linspace(f(x_min_plot), f(x_max_plot), 7))
x_ticks = torch.linspace(0, grid_size_abs, 7) - 0.5
plot_y_ticks = torch.linspace(y_min_plot, y_max_plot, 7)
y_ticks = torch.linspace(0, grid_size_ord, 7) - 0.5

plt.gca().set_xticks(x_ticks)
plt.gca().set_xticklabels([f"{val:.0e}" for val in plot_x_ticks] if plot == 'log' else [f"{val:.1f}" for val in plot_x_ticks])
plt.gca().set_yticks(y_ticks)
plt.gca().set_yticklabels([f"{val:.1f}" for val in plot_y_ticks])
plt.gca().set_xlabel("standard deviation")
plt.gca().set_ylabel("mean")

alpha_mean_min = (y_min - y_max_plot) / (y_min_plot - y_max_plot)
alpha_mean_max = (y_max - y_max_plot) / (y_min_plot - y_max_plot)
alpha_std_min = (f(x_min) - f(x_max_plot)) / (f(x_min_plot) - f(x_max_plot))
alpha_std_max = (f(x_max) - f(x_max_plot)) / (f(x_min_plot) - f(x_max_plot))
square_y = [
    grid_size_ord * (1 - alpha_mean_min) - 0.5,
    grid_size_ord * (1 - alpha_mean_max) - 0.5,
    grid_size_ord * (1 - alpha_mean_max) - 0.5,
    grid_size_ord * (1 - alpha_mean_min) - 0.5,
    grid_size_ord * (1 - alpha_mean_min) - 0.5
]
square_x = [
    grid_size_abs * (1 - alpha_std_min) - 0.5,
    grid_size_abs * (1 - alpha_std_min) - 0.5,
    grid_size_abs * (1 - alpha_std_max) - 0.5,
    grid_size_abs * (1 - alpha_std_max) - 0.5,
    grid_size_abs * (1 - alpha_std_min) - 0.5
]
plt.plot(square_x, square_y, 'red', linewidth=1)

plt.show()
