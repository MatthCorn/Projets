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

def VisualizeScenario(save_path):
    from Inter.NetworkGlobalWindowed.SpecialUtils import GetData
    import torch

    param = loadXmlAsObj(os.path.join(save_path, 'param'))
    weight_l = torch.load(os.path.join(save_path, 'WeightL'), weights_only=False)
    weight_f = torch.load(os.path.join(save_path, 'WeightF'), weights_only=False)

    [Input, Output, Masks, _], _ = GetData(
        d_in=param['d_in'],
        n_pulse_plateau=param['n_pulse_plateau'],
        n_sat=param['n_sat'],
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
        size_focus_source=param['len_in_window'] - param['size_tampon_source'],
        size_tampon_source=param['size_tampon_source'],
        size_tampon_target=param['size_tampon_target'],
        size_focus_target=param['len_out_window'] - param['size_tampon_target'],
        parallel=True,
        max_inflight=500,
    )

    from Inter.NetworkGlobalWindowed.Network import TransformerTranslator
    N = TransformerTranslator(param['d_in'], param['d_in'] + 1, d_att=param['d_att'], n_heads=param['n_heads'], n_encoders=param['n_encoder'],
                              n_decoders=param['n_decoder'], widths_embedding=param['widths_embedding'], width_FF=param['width_FF'], len_in=param['len_in'],
                              len_out=param['len_out'], norm=param['norm'], dropout=param['dropout'])
    N.load_state_dict(torch.load(os.path.join(save_path, 'Last_network')))

    InputMask = Masks[:-1]
    WindowMask = Masks[-1]

    Prediction = N(Input, Output, InputMask)[:, :-1, :] * WindowMask

    size_focus_source = param['len_in_window'] - param['size_tampon_source']
    size_tampon_source = param['size_tampon_source']
    size_tampon_target = param['size_tampon_target']
    size_focus_target = param['len_out_window'] - param['size_tampon_target']

    Prediction[:, :, -1] = Prediction[:, :, -1] + (
        torch.arange(Prediction.shape[0]).view(-1, 1) * size_focus_source +
        torch.arange(-size_tampon_target, size_focus_target).view(1, -1)
    ) * WindowMask[:, :, -1]

    Prediction = Prediction[WindowMask.to(bool).squeeze(-1)]

    Output[:, :, -1] = Output[:, :, -1] + (
        torch.arange(Output.shape[0]).view(-1, 1) * size_focus_source +
        torch.arange(-size_tampon_target, size_focus_target).view(1, -1)
    ) * WindowMask[:, :, -1]

    Output = Output[WindowMask.to(bool).squeeze(-1)]

    SourceMask = 1 - Masks[0] - Masks[1]
    SourceMask[:, :size_tampon_source] = 0

    Input = Input[SourceMask.to(bool).squeeze(-1)]

    df = param['sensitivity']
    range_plot = param['len_in'] + param['n_pulse_plateau']
    f_min = Input[:, 0].min() - 5 * df
    f_max = Input[:, 0].max() + 5 * df
    l_std = Input[:, 1].std()

    from matplotlib import colors
    from matplotlib.patches import Rectangle
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    L = Input.tolist()
    enum_L_sorted = sorted(enumerate(L), key=lambda x: x[1])
    for i, vector in enum_L_sorted:
        T1 = i
        T2 = T1 + vector[-1]
        F = vector[0]
        N = 0.5 * np.tanh(vector[1]/l_std) + 1

        r, g, b, a = value_to_rgb(N)

        rect = Rectangle((T1, F - df),  # coin bas gauche
                     T2 - T1,       # largeur
                     2 * df,        # hauteur
                     facecolor=(r, g, b, 0.8),
                     edgecolor='k',
                     linewidth=0.3)
        ax1.add_patch(rect)

    R = Output.tolist()
    R.sort(key=lambda x: x[1])
    for vector in R:
        T1 = vector[-1]
        T2 = T1 + vector[-2]
        F = vector[0]
        N = 0.5 * np.tanh(vector[1]/l_std) + 1

        r, g, b, a = value_to_rgb(N)

        rect = Rectangle((T1, F - df),  # coin bas gauche
                         T2 - T1,  # largeur
                         2 * df,  # hauteur
                         facecolor=(r, g, b, 0.8),
                         edgecolor='k',
                         linewidth=0.3)
        ax2.add_patch(rect)

    L = Prediction.tolist()
    L.sort(key=lambda x: x[1])
    for vector in L:
        T1 = vector[-1]
        T2 = T1 + vector[-2]
        F = vector[0]
        N = 0.5 * np.tanh(vector[1]/l_std) + 1

        r, g, b, a = value_to_rgb(N)

        rect = Rectangle((T1, F - df),  # coin bas gauche
                         T2 - T1,  # largeur
                         2 * df,  # hauteur
                         facecolor=(r, g, b, 0.8),
                         edgecolor='k',
                         linewidth=0.3)
        ax3.add_patch(rect)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cmap = plt.get_cmap('plasma')
    norm = colors.Normalize(vmin=0, vmax=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sm, cax=cax)

    ax1.set_ylabel('fréquence')
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(-2, range_plot)
        ax.set_ylim(f_min, f_max)
        ax.set_xlabel('temps')

    ax2.set_yticks([])
    ax3.set_yticks([])


    ax1.set_title("Source sequence")
    ax2.set_title("Target sequence")
    ax3.set_title("Predicted sequence")

    plt.tight_layout()

    def on_lim_changed(event_ax):
        global updating
        if updating:
            return  # on est déjà en train de mettre à jour, on sort

        updating = True
        try:
            xlim = event_ax.get_xlim()
            ylim = event_ax.get_ylim()

            for ax in (ax1, ax2, ax3):
                if ax is not event_ax:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
            event_ax.figure.canvas.draw_idle()
        finally:
            updating = False

    # Attacher l'événement
    for ax in (ax1, ax2, ax3):
        ax.callbacks.connect('xlim_changed', on_lim_changed)
        ax.callbacks.connect('ylim_changed', on_lim_changed)

    plt.show()

if __name__ == '__main__':
    save_path = r'C:\Users\matth\Documents\Python\Projets\Inter\NetworkGlobalWindowed\Save\2025-07-08__16-59'

    PlotError(save_path)

    VisualizeScenario(save_path)

    # PathToGIF(save_path)
