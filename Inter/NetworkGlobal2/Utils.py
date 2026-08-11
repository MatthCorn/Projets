import torch
from Inter.Model.DataMaker import GetData as GD

def GetData(d_in, n_pulse_plateau, n_sat, n_mes, len_in, len_out, n_data_training, n_data_validation=1000, sensitivity=0.1,
            weight_f=None, weight_l=None, bias='none', std_min=1., std_max=5., mean_min=-10., mean_max=10.,
            distrib='log', plot=False, save_path=None, parallel=False):

    data = GD(d_in, n_pulse_plateau, n_sat, n_mes, len_in, len_out, n_data_training, n_data_validation=n_data_validation,
                   sensitivity=sensitivity, weight_f=weight_f, weight_l=weight_l, bias=bias, std_min=std_min,
                   std_max=std_max, mean_min=mean_min, mean_max=mean_max, distrib=distrib,
                   save_path=save_path,  parallel=parallel, type='complete')

    if plot:
        raise ValueError

    else:
        data_training, data_validation = data
        input_data_t, output_data_t, [add_mask_t, mult_mask_t], std_t = data_training
        input_data_v, output_data_v, [add_mask_v, mult_mask_v], std_v = data_validation
        return ([*modif(input_data_t, output_data_t), [add_mask_t, mult_mask_t], std_t],
                [*modif(input_data_v, output_data_v), [add_mask_v, mult_mask_v], std_v])

def modif(input_data, output_data):
    ToA_in = torch.range(0, input_data.shape[1] - 1).unsqueeze(0).unsqueeze(2)
    dToA_out = torch.range(0, output_data.shape[1] - 1).unsqueeze(0).unsqueeze(2)

    input_data = torch.cat([input_data, ToA_in.expand(len(input_data), -1, -1)], dim=2)
    output_data[..., -1:] = (dToA_out - output_data[..., -1:])
    return (input_data, output_data)
