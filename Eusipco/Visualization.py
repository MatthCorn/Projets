import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from Tools.XMLTools import loadXmlAsObj

import os

def MakeGIF(PlottingData, NData, training_strategy, frac, distrib, save_path):
    if distrib == 'uniform':
        f = lambda x: x
        g = lambda x: x
    elif distrib == 'log':
        f = lambda x: np.log(x)
        g = lambda x: np.exp(x)

    frames = len(PlottingData[0])

    mean_min = min([window['mean'][0] for window in training_strategy])
    mean_max = max([window['mean'][1] for window in training_strategy])
    std_min = min([window['std'][0] for window in training_strategy])
    std_max = max([window['std'][1] for window in training_strategy])


    plot_x_ticks = g(np.linspace(f(std_min), f(std_max), 5))
    x_ticks = np.linspace(0, NData, 5)
    plot_y_ticks = np.linspace(mean_min, mean_max, 7)
    y_ticks = np.linspace(0, NData, 7)

    # Prepare the figure
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # Update function for FuncAnimation
    def update(frame):
        ax[0].clear()
        ax[1].clear()

        ax[0].set_xticks(x_ticks - 0.5)
        ax[0].set_xticklabels([f"{val:.1f}" for val in plot_x_ticks] if distrib == 'uniform' else [f"{val:.0e}" for val in plot_x_ticks])
        ax[0].set_yticks(y_ticks - 0.5)
        ax[0].set_yticklabels([f"{val:.1f}" for val in plot_y_ticks])

        ax[1].set_xticks(x_ticks - 0.5)
        ax[1].set_xticklabels([f"{val:.1f}" for val in plot_x_ticks] if distrib == 'uniform' else [f"{val:.0e}" for val in plot_x_ticks])
        ax[1].set_yticks(y_ticks - 0.5)
        ax[1].set_yticklabels([f"{val:.1f}" for val in plot_y_ticks])

        ax[0].set_xlabel("standard deviation")
        ax[0].set_ylabel("mean")

        ax[1].set_xlabel("standard deviation")
        ax[1].set_ylabel("mean")

        ax[0].set_title("error :" + str(frame))
        ax[1].set_title("accuracy :" + str(frame))
        error_im = PlottingData[0][frame]
        perf_im = PlottingData[1][frame]
        im0 = ax[0].imshow(error_im, cmap="cool", vmin=0, vmax=1)
        im1 = ax[1].imshow(perf_im, cmap="cool", vmin=0, vmax=1)

        current_strat = training_strategy[int(frame/frames*frac * len(training_strategy))]
        current_mean_min = current_strat['mean'][0]
        current_mean_max = current_strat['mean'][1]
        current_std_min = current_strat['std'][0]
        current_std_max = current_strat['std'][1]
        alpha_mean_min = (current_mean_min - mean_max) / (mean_min - mean_max)
        alpha_mean_max = (current_mean_max - mean_max) / (mean_min - mean_max)
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

def PathToGIF(save_path):
    error = loadXmlAsObj(os.path.join(save_path, 'error'))
    param = loadXmlAsObj(os.path.join(save_path, 'param'))
    distrib = param['plot_distrib'] if 'plot_distrib' in param.keys() else param['distrib']
    training_strategy = param['training_strategy']
    PlottingData = [error['PlottingError'], error['PlottingPerf']]
    NData = len(PlottingData[0][0])
    frac = len(error['TrainingError']) / param['n_iter']

    MakeGIF(PlottingData, NData, training_strategy, frac, distrib, save_path)

def PlotError(save_path):
    error = loadXmlAsObj(os.path.join(save_path, 'error'))
    TrainingError = error['TrainingError']
    ValidationError = error['ValidationError']
    TrainingPerf = error['TrainingPerf']
    ValidationPerf = error['ValidationPerf']

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(TrainingError, 'r', label="Training")
    ax1.plot(ValidationError, 'b', label="Validation")
    ax1.plot([1.] * len(ValidationError), 'black')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right')
    ax1.set_title("Erreur")

    ax2.plot(TrainingPerf, 'r', label="Training")
    ax2.plot(ValidationPerf, 'b', label="Validation")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right')
    ax2.set_title("Accuracy")

    fig.tight_layout(pad=1.0)

    plt.show()

def PlotEvalParam(save_path):
    error_path = os.path.join(save_path, 'error')
    param_path = os.path.join(save_path, 'param')
    error = loadXmlAsObj(error_path)
    param = loadXmlAsObj(param_path)

    if param['eval']['spacing'] == 'log':
        f = np.log
        g = np.exp
    elif param['eval']['spacing'] == 'uniform':
        f = lambda x: x
        g = lambda x: x

    lbd = g(np.linspace(f(param['eval']['multiplier'][0]), f(param['eval']['multiplier'][1]),param['eval']['n_points_reg'], endpoint=True))
    x = param[param['eval']['param']] * lbd

    upper_error = np.array(error['FinalErrorValidation']) + np.array(error['NoiseErrorValidation'])
    middle_error = np.array(error['FinalErrorValidation'])
    lower_error = np.array(error['FinalErrorValidation']) - np.array(error['NoiseErrorValidation'])

    upper_perf = np.array(error['FinalPerfValidation']) + np.array(error['NoisePerfValidation'])
    middle_perf = np.array(error['FinalPerfValidation'])
    lower_perf = np.array(error['FinalPerfValidation']) - np.array(error['NoisePerfValidation'])

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(middle_error, 'r')
    ax1.plot(upper_error, 'r')
    ax1.plot(lower_error, 'r')
    ax1.fill_between([i for i in range(len(middle_error))], lower_error, upper_error, color='r', alpha=0.5)
    ax1.set_xticks([i for i in range(len(x))])
    ax1.set_xticklabels([f"{val:.0e}" for val in x])
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel(param['eval']['param'])
    ax1.set_title("Erreur")
    ax1.set_box_aspect(1)

    ax2.plot(middle_perf, 'b')
    ax2.plot(upper_perf, 'b')
    ax2.plot(lower_perf, 'b')
    ax2.set_xticks([i for i in range(len(x))])
    ax2.fill_between([i for i in range(len(middle_perf))], upper_perf, lower_perf, color='b', alpha=0.5)
    ax2.set_ylim(bottom=0)
    ax2.set_xticklabels([f"{val:.0e}" for val in x])
    ax2.set_xlabel(param['eval']['param'])
    ax2.set_title("Accuracy")
    ax2.set_box_aspect(1)

    fig.tight_layout(pad=1.0)

    plt.show()

if __name__ == '__main__':
    save_path = r'C:\Users\Matth\Documents\Projets\Eusipco\Save\eval_param_2025-03-06__09-49'

    PlotEvalParam(save_path)