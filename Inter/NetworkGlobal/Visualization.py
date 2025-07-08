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

def VisualizeScenario(save_path):
    from Inter.Model.DataMaker import GetData
    import torch

    param = loadXmlAsObj(os.path.join(save_path, 'param'))
    weight_l = torch.load(os.path.join(save_path, 'WeightL'), weights_only=False)
    weight_f = torch.load(os.path.join(save_path, 'WeightF'), weights_only=False)

    [Input, Output, Masks, _], _ = GetData(
        d_in=param['d_in'],
        n_pulse_plateau=param["n_pulse_plateau"],
        n_sat=param["n_sat"],
        len_in=param["len_in"],
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
        plot=False,
        type='complete',
        parallel=True
    )

    from Inter.NetworkGlobal.Network import TransformerTranslator
    N = TransformerTranslator(param['d_in'], param['d_in'] + 1, d_att=param['d_att'], n_heads=param['n_heads'], n_encoders=param['n_encoder'],
                              n_decoders=param['n_decoder'], widths_embedding=param['widths_embedding'], width_FF=param['width_FF'], len_in=param['len_in'],
                              len_out=param['len_out'], norm=param['norm'], dropout=param['dropout'])
    N.load_state_dict(torch.load(os.path.join(save_path, 'Last_network')))
    print(sum(p.numel() for p in N.parameters() if p.requires_grad))
    Prediction = N(Input, Output, Masks)[:, :-1, :]

    df = param['sensitivity']
    range_plot = param['len_in'] + param['n_pulse_plateau']
    f_min = Input[:, :, 0].min() - 5 * df
    f_max = Input[:, :, 0].max() + 5 * df
    l_std = Input[0, :, 1].std()

    from matplotlib import colors
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={'projection': '3d'})

    L = Input[0].tolist()
    for i, vector in enumerate(L):
        T1 = i
        T2 = T1 + vector[-1]
        F = vector[0]
        N = 0.5 * np.tanh(vector[1]/l_std) + 1
        sommets = [
            [T1, F, N],
            [T2, F, N],
            [T1, F - df, 0],
            [T1, F + df, 0],
            [T2, F + df, 0],
            [T2, F - df, 0]
        ]

        surf = np.array([[sommets[0], sommets[0], sommets[0], sommets[0]],
                         [sommets[0], sommets[2], sommets[3], sommets[0]],
                         [sommets[1], sommets[5], sommets[4], sommets[1]],
                         [sommets[1], sommets[1], sommets[1], sommets[1]]])

        # Plot the surface
        r, g, b, a = value_to_rgb(N)
        ax1.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))

    R = Output[0][:Masks[0][0, :, 0].tolist().index(1.)].tolist()
    for i, vector in enumerate(R):
        T1 = i - vector[-1]
        T2 = T1 + vector[-2]
        F = vector[0]
        N = 0.5 * np.tanh(vector[1]/l_std) + 1

        sommets = [
            [T1, F, N],
            [T2, F, N],
            [T1, F - df, 0],
            [T1, F + df, 0],
            [T2, F + df, 0],
            [T2, F - df, 0]
        ]

        surf = np.array([[sommets[0], sommets[0], sommets[0], sommets[0]],
                         [sommets[0], sommets[2], sommets[3], sommets[0]],
                         [sommets[1], sommets[5], sommets[4], sommets[1]],
                         [sommets[1], sommets[1], sommets[1], sommets[1]]])

        # Plot the surface
        r, g, b, a = value_to_rgb(N)
        ax2.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))

    L = Prediction[0][:Masks[0][0, :, 0].tolist().index(1.)].tolist()
    for i, vector in enumerate(L):
        T1 = i - vector[-1]
        T2 = T1 + vector[-2]
        F = vector[0]
        N = 0.5 * np.tanh(vector[1]/l_std) + 1

        sommets = [
            [T1, F, N],
            [T2, F, N],
            [T1, F - df, 0],
            [T1, F + df, 0],
            [T2, F + df, 0],
            [T2, F - df, 0]
        ]

        surf = np.array([[sommets[0], sommets[0], sommets[0], sommets[0]],
                         [sommets[0], sommets[2], sommets[3], sommets[0]],
                         [sommets[1], sommets[5], sommets[4], sommets[1]],
                         [sommets[1], sommets[1], sommets[1], sommets[1]]])

        # Plot the surface
        r, g, b, a = value_to_rgb(N)
        ax3.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))

    # Define the colormap and normalization (vmin, vmax)
    cmap = plt.get_cmap('plasma')  # You can choose 'plasma', 'coolwarm', etc.
    norm = colors.Normalize(vmin=0, vmax=2)  # Define the range for the colorbar

    # Create a ScalarMappable to be used for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Optional, required in some cases to avoid warnings

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cax1 = inset_axes(ax1, width="5%", height="72%", loc='right', borderpad=0.05)
    fig.colorbar(sm, cax=cax1)
    cax2 = inset_axes(ax2, width="5%", height="72%", loc='right', borderpad=0.05)
    fig.colorbar(sm, cax=cax2)
    cax3 = inset_axes(ax3, width="5%", height="72%", loc='right', borderpad=0.05)
    fig.colorbar(sm, cax=cax3)
    plt.tight_layout()

    ax1.set_xlim(-2, range_plot)
    ax1.set_ylim(f_min, f_max)
    ax1.set_zlim(0, 2)
    ax1.set_box_aspect([2, 2, 0.5])
    ax1.zaxis.set_ticks([])  # Remove z-axis ticks
    ax1.zaxis.set_ticklabels([])  # Remove z-axis tick labels
    ax1.zaxis.label.set_visible(False)  # Hide z-axis label
    ax1.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax1.set_xlabel('temps')
    ax1.set_ylabel('fréquence')
    ax1.set_proj_type('ortho')
    ax1.view_init(elev=90, azim=-90)
    ax1.set_xlim(-2, range_plot)
    ax2.set_ylim(f_min, f_max)
    ax2.set_zlim(0, 2)
    ax2.set_box_aspect([2, 2, 0.5])
    ax2.zaxis.set_ticks([])  # Remove z-axis ticks
    ax2.zaxis.set_ticklabels([])  # Remove z-axis tick labels
    ax2.zaxis.label.set_visible(False)  # Hide z-axis label
    ax2.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax2.set_xlabel('temps')
    ax2.set_ylabel('fréquence')
    ax2.set_proj_type('ortho')
    ax2.view_init(elev=90, azim=-90)
    ax1.set_xlim(-2, range_plot)
    ax3.set_ylim(f_min, f_max)
    ax3.set_zlim(0, 2)
    ax3.set_box_aspect([2, 2, 0.5])
    ax3.zaxis.set_ticks([])  # Remove z-axis ticks
    ax3.zaxis.set_ticklabels([])  # Remove z-axis tick labels
    ax3.zaxis.label.set_visible(False)  # Hide z-axis label
    ax3.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax3.set_xlabel('temps')
    ax3.set_ylabel('fréquence')
    ax3.set_proj_type('ortho')
    ax3.view_init(elev=90, azim=-90)

    ax1.set_title("Source sequence")
    ax2.set_title("Target sequence")
    ax3.set_title("Predicted sequence")

    plt.show()

if __name__ == '__main__':
    save_path = r'C:\Users\Matth\Documents\Projets\Inter\NetworkGlobal\Save\2025-07-07__14-48'

    PlotError(save_path)

    VisualizeScenario(save_path)

    # PathToGIF(save_path)
