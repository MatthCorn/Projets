import numpy as np
import torch
from FakeDigitalTwin.Pulse import Pulse
from FakeDigitalTwin.Simulator import DigitalTwin
import matplotlib.pyplot as plt
from matplotlib import colors
from Tools.XMLTools import loadXmlAsObj
import os

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')

Param = {'Fe_List':[5.1, 5, 4.9, 4.8],
        'Duree_max_impulsion':4,
        'Seuil_mono':10,
        'Seuil_harmo':8,
        'Seuil_IM':8,
        'Seuil_sensi_traitement':6,
        'Seuil_sensi':1,
        'Contraste_geneur':0.2,
        'Nint':500,
        'Contraste_geneur_2':1,
        'M1_aveugle':2,
        'M2_aveugle':2,
        'M_local':5,
        'N_DetEl':12,
        'Seuil_ecart_freq':5e-3,
        'Duree_maintien_max':0.2,
        'N_mesureurs_max':8,
        'PDW_tries':0
}

def value_to_rgb(value, min_val=0, max_val=5.5, colormap='plasma'):
    # Normalize the value between 0 and 1
    normalized_value = (value - min_val) / (max_val - min_val)

    # Clip the normalized value to ensure it stays within [0, 1]
    normalized_value = np.clip(normalized_value, 0, 1)

    # Get the colormap
    cmap = plt.get_cmap(colormap)

    # Map the normalized value to an RGB color
    rgb = cmap(normalized_value)  # Returns an RGBA tuple, we need the RGB part

    return rgb

def Plot_inplace(Simulateur, device, Param=Param, seed=None, rec=None):
    if seed is not None:
        np.random.seed(seed)

    len_in = Simulateur.len_in
    len_out = Simulateur.len_out

    t_max = int(len_in * 1.06 / 4)
    TOA = t_max * np.sort(np.random.random(size=len_in))
    LIcourte = np.random.uniform(0.1, 1, int(len_in * 1)) #0.97
    LILongue = np.random.uniform(15, 20, len_in - int(len_in * 1))
    LI = np.concatenate([LILongue, LIcourte])
    np.random.shuffle(LI)
    Level = np.random.gamma(shape=2, scale=1.725, size=len_in)
    dF = 0.01 * (2 * np.random.random(size=len_in) - 1)
    FreqMoy = 0.3 * np.random.random(size=len_in) + 9.5
    FreqStart = FreqMoy + dF
    FreqEnd = FreqMoy - dF

    AntP = [Pulse(TOA=round(TOA[k], 3), LI=round(LI[k], 3), Level=round(Level[k], 3), FreqStart=round(FreqStart[k], 3),
                  FreqEnd=round(FreqEnd[k], 3), Id=k) for k in range(len_in)]


    DT = DigitalTwin(Param)
    DT.forward(AntPulses=AntP)

    input = [[pulse.TOA, pulse.LI, pulse.Level, pulse.FreqStart, pulse.FreqEnd] for pulse in AntP]
    output = [[pulse['TOA'], pulse['LI'], pulse['Level'], pulse['FreqMin'], pulse['FreqMax']] for pulse in DT.PDWs]
    real_len_out = len(output)
    output += [[0.] * 5] * (len_out - real_len_out)
    arange = torch.arange(len_out + 1).unsqueeze(0)
    add_mask = torch.tensor((real_len_out + 1) == arange, dtype=torch.float, device=device).unsqueeze(-1)
    mult_mask = torch.tensor((real_len_out + 1) >= arange, dtype=torch.float, device=device).unsqueeze(-1)
    input = torch.tensor(input, dtype=torch.float, device=device).unsqueeze(0)
    output = torch.tensor(output, dtype=torch.float, device=device).unsqueeze(0)

    # prediction = Simulateur(input, output, [add_mask, mult_mask])[:, :-1, :].detach()
    if rec is None:
        rec_prediction, end = Simulateur.rec_forward(input)
        print(end)
        print(rec_prediction.shape)
        rec_prediction = rec_prediction[end > end.max()/3].unsqueeze(0)
        print(rec_prediction.shape)
        prediction = Simulateur(input, output, [add_mask, mult_mask])[:, :-1, :].detach()

    elif rec:
        prediction, end = Simulateur.rec_forward(input)
        print(end)
        print(prediction.shape)
        rec_prediction = prediction[end > end.max()/3].unsqueeze(0)
        print(prediction.shape)
    else:
        prediction = Simulateur(input, output, [add_mask, mult_mask])[:, :-1, :].detach()

    df = Param['M_local'] * Param['Fe_List'][1] / Param['Nint']
    t_max = max(max(input[0, :, 0] + input[0, :, 1]),
                max(output[0, :real_len_out, 0] + output[0, :real_len_out, 1]),
                max(prediction[0, :real_len_out, 0] + prediction[0, :real_len_out, 1]))
    f_min = min(min(input[0, :, 3]), min(output[0, :real_len_out, 3]), min(prediction[0, :real_len_out, 3])) - 3 * df
    f_max = max(max(input[0, :, 3]), max(output[0, :real_len_out, 3]), max(prediction[0, :real_len_out, 3])) + 3 * df
    level_max = max(max(input[0, :, 2]), max(output[0, :real_len_out, 2]), max(prediction[0, :real_len_out, 2]))

    if rec is None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={'projection': '3d'})

    L = AntP
    for pulse in L:
        T1 = pulse.TOA
        T2 = T1 + pulse.LI
        F1 = pulse.FreqStart
        F2 = pulse.FreqEnd
        N = pulse.Level
        sommets = [
            [T1, F1, N],
            [T2, F2, N],
            [T1, F1 - df, 0],
            [T1, F1 + df, 0],
            [T2, F2 + df, 0],
            [T2, F2 - df, 0]
        ]

        surf = np.array([[sommets[0], sommets[0], sommets[0], sommets[0]],
                         [sommets[0], sommets[2], sommets[3], sommets[0]],
                         [sommets[1], sommets[5], sommets[4], sommets[1]],
                         [sommets[1], sommets[1], sommets[1], sommets[1]]])

        # Plot the surface
        r, g, b, a = value_to_rgb(N, max_val=level_max)
        ax1.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))

    R = DT.PDWs
    for pulse in R:
        T1 = pulse['TOA']
        T2 = T1 + pulse['LI']
        F = (pulse['FreqMin'] + pulse['FreqMax']) / 2
        N = pulse['Level']

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
        r, g, b, a = value_to_rgb(N, max_val=level_max)
        ax2.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))

    L = prediction[0]
    for pulse in L:
        T1 = pulse[0]
        T2 = T1 + pulse[1]
        F = (pulse[3] + pulse[4]) / 2
        N = pulse[2]
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
        r, g, b, a = value_to_rgb(N, max_val=level_max)
        ax3.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))

    if rec is None:
        L = rec_prediction[0]
        for pulse in L:
            T1 = pulse[0]
            T2 = T1 + pulse[1]
            F = (pulse[3] + pulse[4]) / 2
            N = pulse[2]
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
            r, g, b, a = value_to_rgb(N, max_val=level_max)
            ax4.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))

    # Define the colormap and normalization (vmin, vmax)
    cmap = plt.get_cmap('plasma')  # You can choose 'plasma', 'coolwarm', etc.
    norm = colors.Normalize(vmin=0, vmax=5.5)  # Define the range for the colorbar

    # Create a ScalarMappable to be used for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Optional, required in some cases to avoid warnings

    # Add the colorbar to the figure without any plot
    fig.colorbar(sm, ax=ax2, shrink=0.4)
    fig.colorbar(sm, ax=ax1, shrink=0.4)
    fig.colorbar(sm, ax=ax3, shrink=0.4)
    if rec is None:
        fig.colorbar(sm, ax=ax4, shrink=0.4)

    ax1.set_xlim(0, t_max)
    ax1.set_ylim(f_min, f_max)
    ax1.set_zlim(0, level_max)
    ax1.set_box_aspect([2, 2, 0.5])
    ax1.zaxis.set_ticks([])  # Remove z-axis ticks
    ax1.zaxis.set_ticklabels([])  # Remove z-axis tick labels
    ax1.zaxis.label.set_visible(False)  # Hide z-axis label
    ax1.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax1.set_xlabel('temps')
    ax1.set_ylabel('fréquence')
    ax1.set_proj_type('ortho')
    ax1.view_init(elev=90, azim=-90)
    ax2.set_xlim(0, t_max)
    ax2.set_ylim(f_min, f_max)
    ax2.set_zlim(0, level_max)
    ax2.set_box_aspect([2, 2, 0.5])
    ax2.zaxis.set_ticks([])  # Remove z-axis ticks
    ax2.zaxis.set_ticklabels([])  # Remove z-axis tick labels
    ax2.zaxis.label.set_visible(False)  # Hide z-axis label
    ax2.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax2.set_xlabel('temps')
    ax2.set_ylabel('fréquence')
    ax2.set_proj_type('ortho')
    ax2.view_init(elev=90, azim=-90)
    ax3.set_xlim(0, t_max)
    ax3.set_ylim(f_min, f_max)
    ax3.set_zlim(0, level_max)
    ax3.set_box_aspect([2, 2, 0.5])
    ax3.zaxis.set_ticks([])  # Remove z-axis ticks
    ax3.zaxis.set_ticklabels([])  # Remove z-axis tick labels
    ax3.zaxis.label.set_visible(False)  # Hide z-axis label
    ax3.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax3.set_xlabel('temps')
    ax3.set_ylabel('fréquence')
    ax3.set_proj_type('ortho')
    ax3.view_init(elev=90, azim=-90)

    ax1.set_title("Impulsions d'entrée")
    ax2.set_title("Impulsions de sortie")
    ax3.set_title("Impulsions prédites")

    if rec is None:
        ax4.set_title("Impulsions prédites recursivement")
        ax4.set_xlim(0, t_max)
        ax4.set_ylim(f_min, f_max)
        ax4.set_zlim(0, level_max)
        ax4.set_box_aspect([2, 2, 0.5])
        ax4.zaxis.set_ticks([])  # Remove z-axis ticks
        ax4.zaxis.set_ticklabels([])  # Remove z-axis tick labels
        ax4.zaxis.label.set_visible(False)  # Hide z-axis label
        ax4.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax4.set_xlabel('temps')
        ax4.set_ylabel('fréquence')
        ax4.set_proj_type('ortho')
        ax4.view_init(elev=90, azim=-90)
    plt.show()

def Plot_afterwards(path_simulateur, path_data, seed=None, rec=False):
    param = loadXmlAsObj(os.path.join(path_simulateur, 'param'))
    from Base.Network import TransformerTranslator

    N = TransformerTranslator(5, 5, d_att=param['d_att'], n_heads=param['n_heads'], n_encoders=param['n_encoder'],
                              n_decoders=param['n_decoder'], widths_embedding=param['widths_embedding'], len_in=param['len_in'],
                              len_out=param['len_out'], norm=param['norm'], dropout=param['dropout'])

    N.load_state_dict(torch.load(os.path.join(path_simulateur, 'Network_weights'), map_location=torch.device('cpu')))

    Param = loadXmlAsObj(os.path.join(path_data, 'arg.xml'))

    device = torch.device('cpu')

    Plot_inplace(N, device, Param=Param, seed=seed, rec=rec)

if __name__ == '__main__':
    good_seeds = [1278, 785, ]
    good_network = ['2024-09-19__19-28', '2024-09-21__16-28', '2024-09-23__16-53', '2024-09-24__13-55', '2024-09-25__09-37', '2024-09-26__12-48']
    Plot_afterwards(
        os.path.join(local, 'Base', 'Save', good_network[-2]),
        os.path.join(local, 'Base', 'Data', 'config0'),
        seed=1278,
        rec=None
    )