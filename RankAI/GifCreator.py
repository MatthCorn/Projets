import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

import os

def MakeGIF(PlottingData, NData, training_strategy, distrib, save_path):
    if distrib == 'uniform':
        f = lambda x: x
        g = lambda x: x
    elif distrib == 'log':
        f = lambda x: np.log(x)
        g = lambda x: np.exp(x)
    elif distrib == 'exp':
        f = lambda x: np.exp(x)
        g = lambda x: np.log(x)

    frames = len(PlottingData[0])

    mean_min = min([window['mean'][0] for window in training_strategy])
    mean_max = max([window['mean'][1] for window in training_strategy])
    std_min = min([window['std'][0] for window in training_strategy])
    std_max = max([window['std'][1] for window in training_strategy])


    log_x_ticks = g(np.linspace(f(std_min), f(std_max), 7))
    x_ticks = np.linspace(0, NData+1, 7)
    log_y_ticks = g(np.linspace(f(mean_min), f(mean_max), 7))
    y_ticks = np.linspace(0, NData+1, 7)

    # Prepare the figure
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # Update function for FuncAnimation
    def update(frame):
        ax[0].clear()
        ax[1].clear()

        ax[0].set_xticks(x_ticks)
        ax[0].set_xticklabels([f"{val:.1f}" for val in log_x_ticks])
        ax[0].set_yticks(y_ticks)
        ax[0].set_yticklabels([f"{val:.1f}" for val in log_y_ticks])

        ax[1].set_xticks(x_ticks)
        ax[1].set_xticklabels([f"{val:.1f}" for val in log_x_ticks])
        ax[1].set_yticks(y_ticks)
        ax[1].set_yticklabels([f"{val:.1f}" for val in log_y_ticks])

        ax[0].set_xlabel("standard deviation")
        ax[0].set_ylabel("mean")

        ax[1].set_xlabel("standard deviation")
        ax[1].set_ylabel("mean")

        ax[0].set_title("error :" + str(frame))
        ax[1].set_title("perf :" + str(frame))
        error_im = PlottingData[0][frame]
        perf_im = PlottingData[1][frame]
        im0 = ax[0].imshow(error_im, cmap="cool", vmin=0, vmax=1)
        im1 = ax[1].imshow(perf_im, cmap="cool", vmin=0, vmax=1)

        current_strat = training_strategy[int(frame/frames * len(training_strategy))]
        current_mean_min = current_strat['mean'][0]
        current_mean_max = current_strat['mean'][1]
        current_std_min = current_strat['std'][0]
        current_std_max = current_strat['std'][1]
        alpha_mean_min = (f(current_mean_min) - f(mean_max)) / (f(mean_min) - f(mean_max))
        alpha_mean_max = (f(current_mean_max) - f(mean_max)) / (f(mean_min) - f(mean_max))
        alpha_std_min = (f(current_std_min) - f(std_max)) / (f(std_min) - f(std_max))
        alpha_std_max = (f(current_std_max) - f(std_max)) / (f(std_min) - f(std_max))
        square_y = [
            NData * (1 - alpha_mean_min) - 0.5,
            NData * (1 - alpha_mean_max) - 0.5,
            NData * (1 - alpha_mean_max) - 0.5,
            NData * (1 - alpha_mean_min) - 0.5,
            NData * (1 - alpha_mean_min) - 0.5
        ]
        square_x = [
            NData * (1 - alpha_std_min) - 0.5,
            NData * (1 - alpha_std_min) - 0.5,
            NData * (1 - alpha_std_max) - 0.5,
            NData * (1 - alpha_std_max) - 0.5,
            NData * (1 - alpha_std_min) - 0.5
        ]
        ax0 = ax[0].plot(square_x, square_y, 'black', linewidth=1)
        ax1 = ax[1].plot(square_x, square_y, 'black', linewidth=1)

        if not hasattr(update, "cbar0"):
            update.cbar0 = plt.colorbar(im0, ax=ax[0])
            update.cbar1 = plt.colorbar(im1, ax=ax[1])
        else:
            update.cbar0.update_normal(im0)
            update.cbar1.update_normal(im1)

        plt.tight_layout()

        return ax0, ax1

    # Create animation
    ani = FuncAnimation(fig, update, frames=frames, blit=False)

    # Save as a GIF
    ani.save(os.path.join(save_path, "RankGIF.gif"), writer=PillowWriter(fps=10))

