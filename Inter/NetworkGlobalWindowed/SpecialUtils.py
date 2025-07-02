import torch
from Inter.Model.DataMaker import GetData as GD

def GetData(d_in, n_pulse_plateau, n_sat, len_in, len_out, n_data_training, n_data_validation=1000, sensitivity=0.1,
            weight_f=None, weight_l=None, bias='none', std_min=1., std_max=5., mean_min=-10., mean_max=10.,
            distrib='log', plot=False, save_path=None, parallel=False, size_tampon_source=10,
            size_focus_source=20, size_tampon_target=15, size_focus_target=30, max_inflight=None):
    window_param = {
        'size_tampon_source': size_tampon_source,
        'size_focus_source': size_focus_source,
        'size_tampon_target': size_tampon_target,
        'size_focus_target': size_focus_target,
    }
    data = GD(d_in, n_pulse_plateau, n_sat, len_in, len_out, n_data_training, n_data_validation=n_data_validation,
                   sensitivity=sensitivity, weight_f=weight_f, weight_l=weight_l, bias=bias, std_min=std_min,
                   std_max=std_max, mean_min=mean_min, mean_max=mean_max, distrib=distrib, plot=plot,
                   save_path=save_path,  parallel=parallel, type='complete', size_tampon_source=size_tampon_source,
                   size_focus_source=size_focus_source, size_tampon_target=size_tampon_target,
                   size_focus_target=size_focus_target, max_inflight=max_inflight)

    if plot:
        input_data, output_data, [add_mask, mult_mask], _ = data
        return window_2(input_data, output_data, mult_mask, add_mask, window_param)

    else:
        data_training, data_validation = data
        input_data_t, output_data_t, [add_mask_t, mult_mask_t], _ = data_training
        input_data_v, output_data_v, [add_mask_v, mult_mask_v], _ = data_validation
        return (window_2(input_data_t, output_data_t, mult_mask_t, add_mask_t, window_param),
                window_2(input_data_v, output_data_v, mult_mask_v, add_mask_v, window_param))

def window(input_data, output_data, mult_mask, add_mask, param):
    size_tampon_source = param['size_tampon_source']
    size_focus_source = param['size_focus_source']
    size_tampon_target = param['size_tampon_target']
    size_focus_target = param['size_focus_target']
    if size_focus_source > input_data.shape[1]:
        raise ValueError('size_focus_window cannot be greater than len_in')
    if size_focus_source <= size_tampon_source:
        raise ValueError('size_focus_window must be greater than size_tampon')
    if size_focus_target > output_data.shape[1]:
        raise ValueError('size_target_window cannot be greater than len_out')
    new_input_data = torch.nn.functional.pad(input_data, (0, 0, size_tampon_source, size_focus_source))
    new_output_data = torch.nn.functional.pad(output_data, (0, 0, size_tampon_target, size_focus_target))

    windowed_input_data = new_input_data.unfold(1, size_tampon_source + size_focus_source, size_focus_source)

    time_of_arrival_window = size_focus_source * torch.arange(1, windowed_input_data.shape[1] + 1)
    time_of_emission = torch.arange(output_data.shape[1]) - output_data[:, :, -1] + output_data[:, :, -2]
    time_of_emission = time_of_emission * mult_mask[:, 1:, 0] + (1 - mult_mask[:, 1:, 0]) * windowed_input_data.shape[1] * size_focus_source
    time_of_emission += (1 - mult_mask[:, :-1, 0])
    time_of_emission = torch.nn.functional.pad(time_of_emission, (0, size_focus_target), value=windowed_input_data.shape[1] * size_focus_source + 1)
    time_of_emission = torch.nn.functional.pad(time_of_emission, (size_tampon_target, 0), value=-1.)
    time_of_emission[:, size_tampon_target - 1] = 0.

    comp = torch.gt(time_of_emission.unsqueeze(-1), time_of_arrival_window.unsqueeze(0).unsqueeze(0))
    arg = torch.nn.functional.pad(comp.to(torch.int).argmax(dim=1), (1, 0), value=size_tampon_target)

    idx = torch.arange(size_tampon_target + size_focus_target).view(1, 1, size_tampon_target + size_focus_target)
    starts = arg[:, :-1].unsqueeze(-1) - size_tampon_target
    window_indices = starts + idx

    batch_size = len(new_output_data)
    batch_idx = torch.arange(batch_size).view(batch_size, 1, 1).expand(-1, windowed_input_data.shape[1], size_tampon_target + size_focus_target)
    windowed_output_data = new_output_data[batch_idx, window_indices]
    mask = torch.nn.functional.pad((1 - mult_mask + add_mask), (0, 0, size_tampon_target, 0), value=-1)
    mask = torch.nn.functional.pad(mask, (0, 0, 0, size_focus_target - 1), value=1)
    mask = mask[batch_idx, window_indices]
    mask = mask.reshape(-1, *mask.shape[2:])
    target_start_mask = (- mask + mask ** 2) / 2
    target_end_mask = (mask + mask ** 2) / 2

    target_window_mult_mask = torch.lt(
        torch.arange(-size_tampon_target, size_focus_target).view(1, 1, size_tampon_target + size_focus_target),
        (arg[:, 1:] - arg[:, :-1]).unsqueeze(-1)
    ).unsqueeze(-1).logical_not().to(torch.float32)

    target_window_add_mask = torch.eq(
        torch.arange(-size_tampon_target, size_focus_target).view(1, 1, size_tampon_target + size_focus_target),
        (arg[:, 1:] - arg[:, :-1]).unsqueeze(-1)
    ).unsqueeze(-1).to(torch.float32)

    target_window_mult_mask = target_window_mult_mask.reshape(-1, *target_window_mult_mask.shape[2:])
    target_window_add_mask = target_window_add_mask.reshape(-1, *target_window_add_mask.shape[2:])

    mask = torch.zeros(*new_input_data.shape[:-1], 1)
    mask[:, :size_tampon_source] = 1
    mask[:, -size_focus_source:] = -1
    mask = mask.unfold(1, size_tampon_source + size_focus_source, size_focus_source).transpose(2, 3)
    mask = mask.reshape(-1, *mask.shape[2:])
    source_start_mask = (mask + mask ** 2) / 2
    source_end_mask = (-mask + mask ** 2) / 2

    windowed_output_data = windowed_output_data.reshape(-1, *windowed_output_data.shape[2:])
    windowed_input_data = windowed_input_data.reshape(-1, *windowed_input_data.shape[2:]).transpose(1, 2)

    return windowed_input_data, windowed_output_data, [source_start_mask, source_end_mask, target_start_mask, target_end_mask, target_window_add_mask, target_window_mult_mask]

def window_2(input_data, output_data, mult_mask, add_mask, param):
    size_tampon_source = param['size_tampon_source']
    size_focus_source = param['size_focus_source']
    size_tampon_target = param['size_tampon_target']
    size_focus_target = param['size_focus_target']
    if size_focus_source > input_data.shape[1]:
        raise ValueError('size_focus_window cannot be greater than len_in')
    if size_focus_source <= size_tampon_source:
        raise ValueError('size_focus_window must be greater than size_tampon')
    if size_focus_target > output_data.shape[1]:
        raise ValueError('size_target_window cannot be greater than len_out')

    new_input_data = torch.nn.functional.pad(input_data, (0, 0, size_tampon_source, size_focus_source))

    windowed_input_data = new_input_data.unfold(1, size_tampon_source + size_focus_source, size_focus_source)

    time_of_arrival_window = size_focus_source * torch.arange(1, windowed_input_data.shape[1] + 1)
    time_of_emission = torch.arange(output_data.shape[1]) - output_data[:, :, -1] + output_data[:, :, -2]
    time_of_emission = time_of_emission * mult_mask[:, 1:, 0] + (1 - mult_mask[:, 1:, 0]) * windowed_input_data.shape[1] * size_focus_source
    time_of_emission += (1 - mult_mask[:, :-1, 0])
    time_of_emission = torch.nn.functional.pad(time_of_emission, (0, size_focus_target), value=windowed_input_data.shape[1] * size_focus_source + 1)
    time_of_emission = torch.nn.functional.pad(time_of_emission, (size_tampon_target, 0), value=-1.)
    time_of_emission[:, size_tampon_target - 1] = 0.

    comp = torch.gt(time_of_emission.unsqueeze(-1), time_of_arrival_window.unsqueeze(0).unsqueeze(0))
    arg = torch.nn.functional.pad(comp.to(torch.int).argmax(dim=1), (1, 0), value=size_tampon_target)

    idx = torch.arange(size_tampon_target + size_focus_target).view(1, 1, size_tampon_target + size_focus_target)
    starts = arg[:, :-1].unsqueeze(-1) - size_tampon_target
    window_indices = starts + idx

    # modification de l'encodage du ToA dans la séquence de sortie
    output_data[:, :, -1] = - output_data[:, :, -1] + torch.arange(output_data.shape[1]).view(1, -1) * mult_mask[:, 1:, 0]
    new_output_data = torch.nn.functional.pad(output_data, (0, 0, size_tampon_target, size_focus_target))

    mask = torch.zeros(*new_input_data.shape[:-1], 1)
    mask[:, :size_tampon_source] = 1
    mask[:, -size_focus_source:] = -1
    mask = mask.unfold(1, size_tampon_source + size_focus_source, size_focus_source).transpose(2, 3)
    mask = mask.reshape(-1, *mask.shape[2:])
    source_start_mask = (mask + mask ** 2) / 2
    source_end_mask = (-mask + mask ** 2) / 2

    batch_size = len(new_output_data)
    batch_idx = torch.arange(batch_size).view(batch_size, 1, 1).expand(-1, windowed_input_data.shape[1], size_tampon_target + size_focus_target)
    windowed_output_data = new_output_data[batch_idx, window_indices]
    mask = torch.nn.functional.pad((1 - mult_mask + add_mask), (0, 0, size_tampon_target, 0), value=-1)
    mask = torch.nn.functional.pad(mask, (0, 0, 0, size_focus_target - 1), value=1)
    mask = mask[batch_idx, window_indices]

    windowed_output_data[:, :, :, -1] = windowed_output_data[:, :, :, -1] - (
        torch.arange(windowed_output_data.shape[1]).view(1, -1, 1) * size_focus_source +
        torch.arange(-size_tampon_target, size_focus_target).view(1, 1, -1)
    ) * (1 - mask[..., 0].abs())

    mask = mask.reshape(-1, *mask.shape[2:])
    target_start_mask = (- mask + mask ** 2) / 2
    target_end_mask = (mask + mask ** 2) / 2

    target_window_mult_mask = torch.lt(
        torch.arange(-size_tampon_target, size_focus_target).view(1, 1, size_tampon_target + size_focus_target),
        (arg[:, 1:] - arg[:, :-1]).unsqueeze(-1)
    ).unsqueeze(-1).logical_not().to(torch.float32)

    target_window_add_mask = torch.eq(
        torch.arange(-size_tampon_target, size_focus_target).view(1, 1, size_tampon_target + size_focus_target),
        (arg[:, 1:] - arg[:, :-1]).unsqueeze(-1)
    ).unsqueeze(-1).to(torch.float32)

    target_window_mult_mask = target_window_mult_mask.reshape(-1, *target_window_mult_mask.shape[2:])
    target_window_add_mask = target_window_add_mask.reshape(-1, *target_window_add_mask.shape[2:])

    windowed_output_data = windowed_output_data.reshape(-1, *windowed_output_data.shape[2:])
    windowed_input_data = windowed_input_data.reshape(-1, *windowed_input_data.shape[2:]).transpose(1, 2)

    # I, O = decode(input_data, plateau_data), decode(input_data, selected_plateau_data)
    # batch_size, seq_len, _ = output_data.shape
    # _, n_sat, dim = O.shape
    #
    # O_reshaped = O.reshape(batch_size, seq_len, n_sat, dim)
    # M = O_reshaped.mean(dim=1, keepdim=True)
    # Std = torch.norm(O_reshaped - M, dim=[1, 2, 3], p=2, keepdim=True) / np.sqrt((seq_len - 1) * n_sat * dim)
    # Std = Std.expand(batch_size, seq_len, 1, 1).reshape(batch_size * seq_len, 1, 1)
    #
    # M = windowed_output_data.mean(dim=1)

    return windowed_input_data, windowed_output_data, [source_start_mask, source_end_mask, target_start_mask, target_end_mask, target_window_add_mask, target_window_mult_mask]
