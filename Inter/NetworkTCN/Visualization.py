import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from Tools.XMLTools import loadXmlAsObj
from Inter.NetworkTCN.Network import MemoryUpdateTCN

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
        error_mem_im = PlottingData[1][frame]
        im0 = ax[0].imshow(error_im, cmap="cool_r", vmin=0, vmax=1)
        im1 = ax[1].imshow(error_mem_im, cmap="cool", vmin=0, vmax=1)

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

        try:
            update.cbar0.remove()
        except:
            pass
        try:
            update.cbar1.remove()
        except:
            pass

        divider0 = make_axes_locatable(ax[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)

        divider1 = make_axes_locatable(ax[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)

        cax0.yaxis.set_ticks_position('right')
        cax1.yaxis.set_ticks_position('right')

        update.cbar0 = plt.colorbar(im0, cax=cax0)
        update.cbar1 = plt.colorbar(im1, cax=cax1)

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
    PlottingData = [error['PlottingError'], error['PlottingMemError']]
    NData = len(PlottingData[0][0])
    frac = len(error['TrainingError']) / param['n_iter']

    MakeGIF(PlottingData, NData, training_strategy, frac, distrib, save_path)

def PlotError(save_path):
    error = loadXmlAsObj(os.path.join(save_path, 'error'))
    TrainingError = error['TrainingError']
    ValidationError = error['ValidationError']
    TrainingErrorNext = error['TrainingErrorNext']
    ValidationErrorNext = error['ValidationErrorNext']

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(TrainingError, 'r', label="Training")
    ax1.plot(ValidationError, 'b', label="Validation")
    ax1.plot([1.] * len(ValidationError), 'black')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right')
    ax1.set_title("Erreur")

    ax2.plot(TrainingErrorNext, 'r', label="Training")
    ax2.plot(ValidationErrorNext, 'b', label="Validation")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right')
    ax2.set_title("Erreur mémoire")

    fig.tight_layout(pad=1.0)

    plt.show()

def value_to_rgb(value, min_val=0, max_val=2, colormap='plasma'):
    # Normalize the value between 0 and 1
    normalized_value = (value - min_val) / (max_val - min_val)

    # Clip the normalized value to ensure it stays within [0, 1]
    normalized_value = np.clip(normalized_value, 0, 1)

    # Get the colormap
    cmap = plt.get_cmap(colormap)

    # Map the normalized value to an RGB color
    rgb = cmap(normalized_value)  # Returns an RGBA tuple, we need the RGB part

    return rgb

updating = False  # flag global pour éviter récursion


def RecursiveGeneration(save_path):
    from Inter.Model.DataMaker import GetData
    from Inter.NetworkTCN.SpecialUtils import PostProcess
    import torch

    param = loadXmlAsObj(os.path.join(save_path, 'param'))
    weight_l = torch.load(os.path.join(save_path, 'WeightL'), weights_only=False)
    weight_f = torch.load(os.path.join(save_path, 'WeightF'), weights_only=False)

    [(Input, Output, Masks, Std), _] = GetData(
        d_in=param['d_in'] - 1,
        n_pulse_plateau=param['n_pulse_plateau'],
        n_sat=param['n_sat'],
        n_mes=param['n_mes'],
        len_in=param['len_in'],
        len_out=param["len_out"],
        n_data_training=1,
        n_data_validation=1,
        sensitivity=param["sensitivity"],
        bias='freq',
        mean_min=min([window["mean"][0] for window in param["training_strategy"]]),
        mean_max=max([window["mean"][1] for window in param["training_strategy"]]),
        std_min=min([window["std"][0] for window in param["training_strategy"]]),
        std_max=max([window["std"][1] for window in param["training_strategy"]]),
        distrib=param["plot_distrib"],
        weight_f=weight_f,
        weight_l=weight_l,
        type='complete',
        parallel=False
    )

    (PInput1, PInput2, POutput, NextMaskInput, NextMaskOutput, OnSequenceMask) = PostProcess(Input, Output, Masks, param['len_in'], param['len_out'], 1)

    N = MemoryUpdateTCN(
        input_dim_1=param['d_in'],
        input_dim_2=param['d_in'],
        hidden_dim=param['d_att'],
        output_dim=param['d_in'],
        tcn_channels=[param['d_att']] * param['n_decoder'],
        kernel_size=3,
        dropout=param['dropout'],
        use_layernorm=True,
    )
    N.load_state_dict(torch.load(os.path.join(save_path, 'Last_network')))

    print(sum(p.numel() for p in N.parameters() if p.requires_grad))

    df = param['sensitivity']
    range_plot = param['len_in'] + param['n_pulse_plateau']
    f_min = Input[:, :, 0].min() - 5 * df
    f_max = Input[:, :, 0].max() + 5 * df
    l_std = Input[0, :, 1].std()

    from matplotlib import colors
    from matplotlib.patches import Rectangle
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # n_pulse_predicted = [x > 0.1 for x in end_list].index(False)
    # L = Prediction[0][:n_pulse_predicted].tolist()
    L = Input[0].tolist()
    for i, vector in enumerate(L):
        T1 = i
        T2 = T1 + vector[-1]
        F = vector[0]
        N = 0.5 * np.tanh(vector[1] / l_std) + 1

        r, g, b, a = value_to_rgb(N)

        rect = Rectangle((T1, F - df),  # coin bas gauche
                         T2 - T1,  # largeur
                         2 * df,  # hauteur
                         facecolor=(r, g, b, 0.8),
                         edgecolor='k',
                         linewidth=0.3)
        ax3.add_patch(rect)
    #
    # GuidedPrediction = GuidedPrediction[0][((1 - NextMaskOutput) * OnSequenceMask)[0, :, 0].to(bool)]
    # TOA_Input = (torch.cumsum(NextMaskInput, dim=1) - 1)[0, :, 0][((1 - NextMaskInput) * OnSequenceMask)[0, :, 0].to(bool)]
    # GuidedPrediction[:, -1] += TOA_Input
    # L = GuidedPrediction.tolist()
    R = Output[0][:Masks[0][0, :, 0].tolist().index(1.)].tolist()
    for i, vector in enumerate(R):
        T1 = i - vector[-1]
        T2 = T1 + vector[-2]
        F = vector[0]
        N = 0.5 * np.tanh(vector[1] / l_std) + 1

        r, g, b, a = value_to_rgb(N)

        rect = Rectangle((T1, F - df),  # coin bas gauche
                         T2 - T1,  # largeur
                         2 * df,  # hauteur
                         facecolor=(r, g, b, 0.8),
                         edgecolor='k',
                         linewidth=0.3)
        ax4.add_patch(rect)

    # GuidedPrediction, _ = N(Input1, Input2, NextMaskInput)

    # end_list = []
    #
    # Prediction = Output
    # for n in range(param['len_out']):
    #     Prediction, is_end = N.recursive_eval(Input, Prediction, n)
    #     Prediction = Prediction[:, :-1, :]
    #     end_list.append(float(is_end))
    #
    # Prediction, is_end = N.recursive_eval(Input, Prediction, n + 1)
    # Prediction = Prediction[:, :-1, :]
    # end_list.append(float(is_end))
    #
    # plt.plot(end_list, 'r')
    # plt.plot(Masks[1][0, :, 0].tolist(), 'b')
    # plt.show()

    PInput = PInput1[0, :, :-1][(NextMaskInput * OnSequenceMask)[0, :, 0].to(bool)]
    L = PInput.tolist()
    for i, vector in enumerate(L):
        T1 = i
        T2 = T1 + vector[-1]
        F = vector[0]
        N = 0.5 * np.tanh(vector[1] / l_std) + 1

        r, g, b, a = value_to_rgb(N)

        rect = Rectangle((T1, F - df),  # coin bas gauche
                         T2 - T1,  # largeur
                         2 * df,  # hauteur
                         facecolor=(r, g, b, 0.8),
                         edgecolor='k',
                         linewidth=0.3)
        ax1.add_patch(rect)

    POutput = POutput[0][((1 - NextMaskOutput) * OnSequenceMask)[0, :, 0].to(bool)]
    TOA_PInput = (torch.cumsum(NextMaskInput, dim=1) - 1)[0, :, 0][((1 - NextMaskInput) * OnSequenceMask)[0, :, 0].to(bool)]
    POutput[:, -1] += TOA_PInput
    R = POutput.tolist()
    for i, vector in enumerate(R):
        T1 = vector[-1]
        T2 = T1 + vector[-2]
        F = vector[0]
        N = 0.5 * np.tanh(vector[1] / l_std) + 1

        r, g, b, a = value_to_rgb(N)

        rect = Rectangle((T1, F - df),  # coin bas gauche
                         T2 - T1,  # largeur
                         2 * df,  # hauteur
                         facecolor=(r, g, b, 0.8),
                         edgecolor='k',
                         linewidth=0.3)
        ax2.add_patch(rect)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cmap = plt.get_cmap('plasma')
    norm = colors.Normalize(vmin=0, vmax=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for ax in (ax2, ax4):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sm, cax=cax)
        ax.set_yticks([])

    ax1.set_ylabel('fréquence')
    ax3.set_ylabel('fréquence')
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlim(-2, range_plot)
        ax.set_ylim(f_min, f_max)
        ax.set_xlabel('temps')


    ax1.set_title("Source sequence")
    ax2.set_title("Target sequence")
    ax3.set_title("Recursive Predicted sequence")
    ax4.set_title("Guided Predicted sequence")

    plt.tight_layout()

    def on_lim_changed(event_ax):
        global updating
        if updating:
            return  # on est déjà en train de mettre à jour, on sort

        updating = True
        try:
            xlim = event_ax.get_xlim()
            ylim = event_ax.get_ylim()

            for ax in (ax1, ax2, ax3, ax4):
                if ax is not event_ax:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
            event_ax.figure.canvas.draw_idle()
        finally:
            updating = False

    # Attacher l'événement
    for ax in (ax1, ax2, ax3, ax4):
        ax.callbacks.connect('xlim_changed', on_lim_changed)
        ax.callbacks.connect('ylim_changed', on_lim_changed)

    plt.show()

if __name__ == '__main__':
    save_path = r'C:\Users\Matth\Documents\Projets\Inter\NetworkTCN\Save\2026-01-21__14-11'

    # PlotError(save_path)

    RecursiveGeneration(save_path)
