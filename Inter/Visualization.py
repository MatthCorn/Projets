import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Update function for FuncAnimation
    def update(frame):
        ax.clear()

        ax.set_xticks(x_ticks - 0.5)
        ax.set_xticklabels([f"{val:.1f}" for val in plot_x_ticks] if distrib == 'uniform' else [f"{val:.0e}" for val in plot_x_ticks])
        ax.set_yticks(y_ticks - 0.5)
        ax.set_yticklabels([f"{val:.1f}" for val in plot_y_ticks])

        ax.set_xlabel("standard deviation")
        ax.set_ylabel("mean")

        ax.set_title("error :" + str(frame))
        error_im = PlottingData[0][frame]
        im0 = ax.imshow(error_im, cmap="cool_r", vmin=0, vmax=1)

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
        ax0 = ax.plot(square_x, square_y, 'black', linewidth=1)

        try:
            update.cbar0.remove()
        except:
            pass

        divider0 = make_axes_locatable(ax)
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)

        cax0.yaxis.set_ticks_position('right')

        update.cbar0 = plt.colorbar(im0, cax=cax0)

        plt.tight_layout()
        return ax0

    # Create animation
    ani = FuncAnimation(fig, update, frames=frames, blit=False)

    # Save as a GIF
    ani.save(os.path.join(save_path, "RankGIF.gif"), writer=PillowWriter(fps=10))

def PathToGIF(save_path):
    error = loadXmlAsObj(os.path.join(save_path, 'error'))
    param = loadXmlAsObj(os.path.join(save_path, 'param'))
    distrib = param['plot_distrib'] if 'plot_distrib' in param.keys() else param['distrib']
    training_strategy = param['training_strategy']
    PlottingData = [error['PlottingError']]
    NData = len(PlottingData[0][0])
    frac = len(error['TrainingError']) / param['n_iter']

    MakeGIF(PlottingData, NData, training_strategy, frac, distrib, save_path)

def PlotError(save_path):
    error = loadXmlAsObj(os.path.join(save_path, 'error'))
    TrainingError = error['TrainingError']
    ValidationError = error['ValidationError']

    fig, ax1 = plt.subplots(1, 1)

    ax1.plot(TrainingError, 'r', label="Training")
    ax1.plot(ValidationError, 'b', label="Validation")
    ax1.plot([1.] * len(ValidationError), 'black')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right')
    ax1.set_title("Erreur")

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

    fig, (ax1, ) = plt.subplots(1, 1)

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

    fig.tight_layout(pad=1.0)

    plt.show()

if __name__ == '__main__':
    save_path = r'C:\Users\matth\Documents\Python\Projets\Inter\NetworkDetection\Save\2025-06-21__16-03'

    PlotError(save_path)

    # PathToGIF(save_path)


    # Tr_path = r'C:\Users\Matth\Documents\Projets\Eusipco\Save\eval_problem_2025-03-08__10-51\error'
    # Cnn_path = r'C:\Users\Matth\Documents\Projets\Eusipco\Save\eval_problem_2025-03-07__21-08\error'
    # Tr_boxplot_path = r'C:\Users\Matth\Documents\Projets\Eusipco\Save\eval_problem_2025-03-12__20-47\error'
    # param_path = r'C:\Users\Matth\Documents\Projets\Eusipco\Save\eval_problem_2025-03-08__10-51\param'
    #
    # import matplotlib.pyplot as plt
    # import matplotlib
    # import numpy as np
    # from Tools.XMLTools import loadXmlAsObj
    # from Eusipco.Boxplot import add_boxplot
    # error_Tr = loadXmlAsObj(Tr_path)
    # error_Cnn = loadXmlAsObj(Cnn_path)
    # error_Tr_boxplot = loadXmlAsObj(Tr_boxplot_path)
    # param = loadXmlAsObj(param_path)
    #
    # if param['distrib'] == 'log':
    #     f = np.log
    #     g = np.exp
    # elif param['distrib'] == 'uniform':
    #     f = lambda x: x
    #     g = lambda x: x
    #
    # lbd = np.flip(g(np.linspace(f(param['lbd']['min']), f(param['lbd']['max']), param['n_points_reg'], endpoint=True)))
    #
    # matplotlib.use('Qt5Agg')
    #
    # upper_error_Tr = np.array(error_Tr['MinError']) + np.array(error_Tr['RightStdMinError'])
    # middle_error_Tr = np.array(error_Tr['MinError'])
    # lower_error_Tr = np.array(error_Tr['MinError']) - np.array(error_Tr['LeftStdMinError'])
    #
    # upper_perf_Tr = np.array(error_Tr['MaxPerf']) + np.array(error_Tr['RightStdMaxPerf'])
    # middle_perf_Tr = np.array(error_Tr['MaxPerf'])
    # lower_perf_Tr = np.array(error_Tr['MaxPerf']) - np.array(error_Tr['LeftStdMaxPerf'])
    #
    # middle_error_Cnn = np.array(error_Cnn['MinError'])
    # middle_perf_Cnn = np.array(error_Cnn['MaxPerf'])
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    #
    # ax1.plot(middle_error_Tr, 'r', label="Transformer")
    # ax1.plot(middle_error_Cnn, 'b', label="CNN")
    # for i in range(10):
    #     add_boxplot(
    #         error_Tr_boxplot['ErrorQ1List'][i],
    #         error_Tr_boxplot['ErrorQ2List'][i],
    #         error_Tr_boxplot['ErrorQ3List'][i],
    #         error_Tr_boxplot['Error_lower_whiskerList'][i],
    #         error_Tr_boxplot['Error_upper_whiskerList'][i],
    #         ax=ax1,
    #         i=i,
    #         color='black'
    #     )
    # ax1.set_ylim(bottom=0)
    # ax1.legend(loc='upper left', framealpha=0.3)
    # ax1.set_title("Error")
    # ax1.set_xticks([i for i in range(len(middle_error_Tr)) if not i%3])
    # ax1.set_xticklabels([f"{lbd[i]:.0e}" for i in range(len(middle_error_Tr)) if not i%3])
    # ax1.set_box_aspect(1)
    #
    # ax2.plot(middle_perf_Tr, 'r', label="Transformer")
    # ax2.plot(middle_perf_Cnn, 'b', label="CNN")
    # for i in range(10):
    #     add_boxplot(
    #         error_Tr_boxplot['PerfQ1List'][i],
    #         error_Tr_boxplot['PerfQ2List'][i],
    #         error_Tr_boxplot['PerfQ3List'][i],
    #         error_Tr_boxplot['Perf_lower_whiskerList'][i],
    #         error_Tr_boxplot['Perf_upper_whiskerList'][i],
    #         ax=ax2,
    #         i=i
    #     )
    # ax2.set_ylim(bottom=0)
    # ax2.legend(loc='lower left', framealpha=0.3)
    # ax2.set_title("Accuracy")
    # ax2.set_xticks([i for i in range(len(middle_error_Tr)) if not i%3])
    # ax2.set_xticklabels([f"{lbd[i]:.0e}" for i in range(len(middle_error_Tr)) if not i%3])
    # ax2.set_box_aspect(1)
    #
    # fig.tight_layout(pad=1.0)
    #
    # plt.show()