import torch
from Inter.Model.Scenario import Simulator, BiasedSimulator, FreqBiasedSimulator
from Tools.XMLTools import loadXmlAsObj, saveObjAsXml
import os
from tqdm import tqdm
import numpy as np
import time

def generate_sample(args):
    # pour les performances
    import psutil, sys

    p = psutil.Process(os.getpid())

    if sys.platform == "win32":
        p.nice(psutil.HIGH_PRIORITY_CLASS)

    import warnings
    warnings.filterwarnings("ignore")  # Désactive les warnings

    """ Génère un échantillon unique basé sur les paramètres. """
    if args[0] == 'none':
        d_in, n_pulse_plateau, n_sat, len_in, len_out, sensitivity, weight_f, weight_l = args[1:]
        S = Simulator(n_pulse_plateau, len_in, d_in - 1, n_sat=n_sat,
                      sensitivity=sensitivity, WeightF=weight_f, WeightL=weight_l)
    elif args[0] == 'freq':
        std, mean, d_in, n_pulse_plateau, n_sat, len_in, len_out, sensitivity, weight_f, weight_l = args[1:]
        S = FreqBiasedSimulator(std, mean, n_pulse_plateau, len_in, d_in - 1, n_sat=n_sat,
                                sensitivity=sensitivity, WeightF=weight_f, WeightL=weight_l)
    elif args[0] == 'all':
        std, mean, d_in, n_pulse_plateau, n_sat, len_in, len_out, sensitivity, weight_f, weight_l = args[1:]
        S = BiasedSimulator(std, mean, n_pulse_plateau, len_in, d_in - 1, n_sat=n_sat,
                            sensitivity=sensitivity, WeightF=weight_f, WeightL=weight_l)
    else:
        raise ValueError

    S.run()

    input_seq = S.L
    plateau_seq = S.P
    selected_plateau_seq = S.D
    len_output_seq = len(S.sensor_simulator.R)
    output_seq = (S.sensor_simulator.R + [[0.] * (d_in + 1)] * (len_out - len_output_seq))[:len_out]

    return input_seq, plateau_seq, selected_plateau_seq, len_output_seq, output_seq

def MakeData(d_in, n_pulse_plateau, n_sat, len_in, len_out, n_data, sensitivity=0.1, weight_f=None, weight_l=None,
             bias='none', std_min=1, std_max=5, mean_min=-10, mean_max=10, distrib='log', plot=False):
    input_data, plateau_data, selected_plateau_data, len_output_data, output_data = [], [], [], [], []
    if bias != 'none':
        if plot:
            spacing = lambda x: np.linspace(0, 1, x)
        else:
            spacing = lambda x: np.random.rand(x)

        if distrib == 'uniform':
            f = lambda x: x
            g = lambda x: x
        elif distrib == 'log':
            f = lambda x: np.log(x)
            g = lambda x: np.exp(x)
        mean_list = (mean_max - mean_min) * spacing(n_data) + mean_min
        std_list = g((f(std_max) - f(std_min)) * spacing(n_data) + f(std_min))

        if plot:
            n_data = n_data ** 2
            std_list, mean_list = np.meshgrid(std_list, mean_list)
            mean_list, std_list = mean_list.flatten(), std_list.flatten()

    for i in tqdm(range(n_data)):
        if bias != 'none':
            mean = mean_list[i]
            std = std_list[i]
            input_seq, plateau_seq, selected_plateau_seq, len_output_seq, output_seq = (
                generate_sample((bias, std, mean, d_in, n_pulse_plateau, n_sat, len_in, len_out, sensitivity, weight_f, weight_l)))
        else:
            input_seq, plateau_seq, selected_plateau_seq, len_output_seq, output_seq = (
                generate_sample((bias, d_in, n_pulse_plateau, n_sat, len_in, len_out, sensitivity, weight_f, weight_l)))

        input_data.append(input_seq)
        plateau_data.append(plateau_seq)
        selected_plateau_data.append(selected_plateau_seq)
        len_output_data.append(len_output_seq)
        output_data.append(output_seq)

    input_data = torch.tensor(input_data, dtype=torch.float)
    output_data = torch.tensor(output_data, dtype=torch.float)
    plateau_data = torch.tensor(plateau_data, dtype=torch.float)
    selected_plateau_data = torch.tensor(selected_plateau_data, dtype=torch.float)

    len_element_output = torch.tensor(len_output_data).unsqueeze(-1)
    arrange = torch.arange(len_out + 1).unsqueeze(0).expand(n_data, -1)
    add_mask = torch.tensor(len_element_output == arrange, dtype=torch.float).unsqueeze(-1)
    mult_mask = torch.tensor(len_element_output >= arrange, dtype=torch.float).unsqueeze(-1)

    return input_data, plateau_data, selected_plateau_data, add_mask, mult_mask, output_data

from concurrent.futures import ProcessPoolExecutor
from collections import deque

def MakeDataParallel(d_in, n_pulse_plateau, n_sat, len_in, len_out, n_data, sensitivity=0.1, weight_f=None, weight_l=None,
                     bias='none', std_min=1, std_max=5, mean_min=-10, mean_max=10, distrib='log', plot=False,
                     executor=None, max_inflight=10000):

    if bias != 'none':
        if plot:
            spacing = lambda x: np.linspace(0, 1, x)
        else:
            spacing = lambda x: np.random.rand(x)

        if distrib == 'uniform':
            f = lambda x: x
            g = lambda x: x
        elif distrib == 'log':
            f = lambda x: np.log(x)
            g = lambda x: np.exp(x)
        mean_list = (mean_max - mean_min) * spacing(n_data) + mean_min
        std_list = g((f(std_max) - f(std_min)) * spacing(n_data) + f(std_min))

        if plot:
            n_data = n_data ** 2
            std_list, mean_list = np.meshgrid(std_list, mean_list)
            mean_list, std_list = mean_list.flatten(), std_list.flatten()

        """ Génère n_data échantillons en parallèle avec ProcessPoolExecutor. """
        args = [(bias, std_list[i], mean_list[i], d_in, n_pulse_plateau, n_sat, len_in, len_out, sensitivity, weight_f, weight_l) for i in range(n_data)]

    else:
        args = [(bias, d_in, n_pulse_plateau, n_sat, len_in, len_out, sensitivity, weight_f, weight_l) for _ in range(n_data)]

    results = []
    inflight = deque()

    with tqdm(total=n_data, desc="Génération") as pbar:
        should_shutdown = False
        if executor is None:
            executor = ProcessPoolExecutor()
            should_shutdown = True

        try:
            it = iter(args)

            # Pré-remplissage
            for _ in range(min(max_inflight, n_data)):
                inflight.append(executor.submit(generate_sample, next(it)))

            while inflight:
                # Attendre qu'une tâche se termine
                done = inflight.popleft()
                results.append(done.result())
                pbar.update(1)

                # En soumettre une nouvelle si dispo
                try:
                    inflight.append(executor.submit(generate_sample, next(it)))
                except StopIteration:
                    continue
        finally:
            if should_shutdown:
                executor.shutdown()


    # Extraction des résultats
    input_data, plateau_data, selected_plateau_data, len_output_data, output_data = zip(*results)

    input_data = torch.tensor(input_data, dtype=torch.float)
    output_data = torch.tensor(output_data, dtype=torch.float)
    plateau_data = torch.tensor(plateau_data, dtype=torch.float)
    selected_plateau_data = torch.tensor(selected_plateau_data, dtype=torch.float)

    len_element_output = torch.tensor(len_output_data).unsqueeze(-1)
    arrange = torch.arange(len_out + 1).unsqueeze(0).expand(n_data, -1)
    add_mask = torch.tensor(len_element_output == arrange, dtype=torch.float).unsqueeze(-1)
    mult_mask = torch.tensor(len_element_output >= arrange, dtype=torch.float).unsqueeze(-1)

    return input_data, plateau_data, selected_plateau_data, add_mask, mult_mask, output_data

def GetData(d_in, n_pulse_plateau, n_sat, len_in, len_out, n_data_training, n_data_validation=1000, sensitivity=0.1,
            weight_f=None, weight_l=None, bias='none', std_min=1., std_max=5., mean_min=-10., mean_max=10.,
            distrib='log', plot=False, save_path=None, parallel=False, type='complete', size_tampon_source=10,
            size_focus_source=20, size_tampon_target=15, size_focus_target=30, max_inflight=None):

    if parallel:
        with ProcessPoolExecutor() as executor:
            return GetDataSecond(d_in, n_pulse_plateau, n_sat, len_in, len_out, n_data_training,
                                 n_data_validation=n_data_validation, sensitivity=sensitivity, weight_f=weight_f,
                                 weight_l=weight_l, bias=bias, std_min=std_min, std_max=std_max, mean_min=mean_min,
                                 mean_max=mean_max, distrib=distrib, plot=plot, save_path=save_path, parallel=parallel,
                                 type=type, size_tampon_source=size_tampon_source, size_focus_source=size_focus_source,
                                 size_tampon_target=size_tampon_target, size_focus_target=size_focus_target,
                                 max_inflight=max_inflight, executor=executor)

    else:
        return GetDataSecond(d_in, n_pulse_plateau, n_sat, len_in, len_out, n_data_training,
                             n_data_validation=n_data_validation, sensitivity=sensitivity, weight_f=weight_f,
                             weight_l=weight_l, bias=bias, std_min=std_min, std_max=std_max, mean_min=mean_min,
                             mean_max=mean_max, distrib=distrib, plot=plot, save_path=save_path, parallel=parallel,
                             type=type, size_tampon_source=size_tampon_source, size_focus_source=size_focus_source,
                             size_tampon_target=size_tampon_target, size_focus_target=size_focus_target,
                             max_inflight=max_inflight)

def GetDataSecond(d_in, n_pulse_plateau, n_sat, len_in, len_out, n_data_training, n_data_validation=1000, sensitivity=0.1,
            weight_f=None, weight_l=None, bias='none', std_min=1., std_max=5., mean_min=-10., mean_max=10.,
            distrib='log', plot=False, save_path=None, parallel=False, type='complete', size_tampon_source=10,
            size_focus_source=20, size_tampon_target=15, size_focus_target=30, max_inflight=None, executor=None):
    make_data = MakeDataParallel if parallel else MakeData

    if bias == 'none':
        kwargs = {'d_in': d_in,
                  'n_pulse_plateau': n_pulse_plateau,
                  'n_sat': n_sat,
                  'len_in': len_in,
                  'len_out': len_out,
                  'sensitivity': sensitivity}
    else:
        kwargs = {'d_in': d_in,
                  'n_pulse_plateau': n_pulse_plateau,
                  'n_sat': n_sat,
                  'len_in': len_in,
                  'len_out': len_out,
                  'sensitivity': sensitivity,
                  'bias': bias,
                  'std_min': std_min,
                  'std_max': std_max,
                  'mean_min': mean_min,
                  'mean_max': mean_max,
                  'distrib': distrib}

    if max_inflight is not None:
        kwargs['max_inflight'] = max_inflight

    if executor is not None:
        kwargs['executor'] = executor

    return_param = {
        'size_tampon_source': size_tampon_source,
        'size_focus_source': size_focus_source,
        'size_tampon_target': size_tampon_target,
        'size_focus_target': size_focus_target,
    }

    if plot or (save_path is None):
        weight_f = weight_f if weight_f is not None else np.array([1., 0.] + [0.] * (d_in - 3))
        weight_f = weight_f / np.linalg.norm(weight_f)
        weight_l = weight_l if weight_l is not None else np.array([0., 1.] + [0.] * (d_in - 3))
        weight_l = weight_l / np.linalg.norm(weight_l)

        if plot:
            data = make_data(n_data=n_data_training, weight_f=weight_f, weight_l=weight_l, plot=True, **kwargs)
            return return_data(data, type, param=return_param)

        data_training = make_data(n_data=n_data_training, weight_f=weight_f, weight_l=weight_l, **kwargs)
        data_validation = make_data(n_data=n_data_validation, weight_f=weight_f, weight_l=weight_l, **kwargs)
        return [return_data(data_training, type, param=return_param), return_data(data_validation, type, param=return_param)]

    try:
        os.mkdir(save_path)
    except:
        pass

    for file in os.listdir(save_path):
        try:
            kwargs_file = loadXmlAsObj(os.path.join(save_path, file, 'kwargs.xml'))
        except:
            continue

        if kwargs_file == {k: v for k, v in kwargs.items() if k != 'executor'}:
            weight_l = np.load(os.path.join(save_path, file, 'weight_l.npy'))
            weight_f = np.load(os.path.join(save_path, file, 'weight_f.npy'))

            n_data = {'training': n_data_training, 'validation': n_data_validation}
            output = []
            for phase in  n_data.keys():

                input_data_ = torch.load(os.path.join(save_path, file, phase + '_input_data'))
                plateau_data_ = torch.load(os.path.join(save_path, file, phase + '_plateau_data'))
                selected_plateau_data_ = torch.load(os.path.join(save_path, file, phase + '_selected_plateau_data'))
                add_mask_ = torch.load(os.path.join(save_path, file, phase + '_add_mask'))
                mult_mask_ = torch.load(os.path.join(save_path, file, phase + '_mult_mask'))
                output_data_ = torch.load(os.path.join(save_path, file, phase + '_output_data'))

                if len(input_data_) < n_data[phase]:
                    input_data, plateau_data, selected_plateau_data, add_mask, mult_mask, output_data  = (
                        make_data(n_data=n_data[phase] - len(input_data_), weight_f=weight_f, weight_l=weight_l, **kwargs))

                    input_data_ = torch.cat((input_data_, input_data), dim=0)
                    plateau_data_ = torch.cat((plateau_data_, plateau_data), dim=0)
                    selected_plateau_data_ = torch.cat((selected_plateau_data_, selected_plateau_data), dim=0)
                    add_mask_ = torch.cat((add_mask_, add_mask), dim=0)
                    mult_mask_ = torch.cat((mult_mask_, mult_mask), dim=0)
                    output_data_ = torch.cat((output_data_, output_data), dim=0)

                    torch.save(input_data_, os.path.join(save_path, file, phase + '_input_data'))
                    torch.save(plateau_data_, os.path.join(save_path, file, phase + '_plateau_data'))
                    torch.save(selected_plateau_data_, os.path.join(save_path, file, phase + '_selected_plateau_data'))
                    torch.save(add_mask_, os.path.join(save_path, file, phase + '_add_mask'))
                    torch.save(mult_mask_, os.path.join(save_path, file, phase + '_mult_mask'))
                    torch.save(output_data_, os.path.join(save_path, file, phase + '_output_data'))

                output.append(list(return_data((input_data_[:n_data[phase]],
                                                plateau_data_[:n_data[phase]],
                                                selected_plateau_data_[:n_data[phase]],
                                                add_mask_[:n_data[phase]],
                                                mult_mask_[:n_data[phase]],
                                                output_data_[:n_data[phase]]
                                                ), type, param=return_param)))

            return output

    attempt = 0
    while True:
        file = f"config({attempt})"

        try:
            os.makedirs(os.path.join(save_path, file), exist_ok=False)
            break
        except FileExistsError:
            attempt += 1
            time.sleep(0.1)

    weight_f = weight_f if weight_f is not None else np.array([1., 0.] + [0.] * (d_in - 3))
    weight_f = weight_f / np.linalg.norm(weight_f)
    weight_l = weight_l if weight_l is not None else np.array([0., 1.] + [0.] * (d_in - 3))
    weight_l = weight_l / np.linalg.norm(weight_l)
    np.save(os.path.join(save_path, file, 'weight_f'), weight_f)
    np.save(os.path.join(save_path, file, 'weight_l'), weight_l)
    saveObjAsXml({k: v for k, v in kwargs.items() if k != 'executor'}, os.path.join(save_path, file, 'kwargs.xml'))

    n_data = {'training': n_data_training, 'validation': n_data_validation}
    output = []
    for phase in n_data.keys():

        input_data, plateau_data, selected_plateau_data, add_mask, mult_mask, output_data = (
            make_data(n_data=n_data[phase], weight_f=weight_f, weight_l=weight_l, **kwargs))



        torch.save(input_data, os.path.join(save_path, file, phase + '_input_data'))
        torch.save(plateau_data, os.path.join(save_path, file, phase + '_plateau_data'))
        torch.save(selected_plateau_data, os.path.join(save_path, file, phase + '_selected_plateau_data'))
        torch.save(add_mask, os.path.join(save_path, file, phase + '_add_mask'))
        torch.save(mult_mask, os.path.join(save_path, file, phase + '_mult_mask'))
        torch.save(output_data, os.path.join(save_path, file, phase + '_output_data'))

        output.append(list(return_data((input_data,
                                        plateau_data,
                                        selected_plateau_data,
                                        add_mask,
                                        mult_mask,
                                        output_data
                                        ), type, param=return_param)))

    return output

def return_data(data, type, param=None):
    input_data, plateau_data, selected_plateau_data, add_mask, mult_mask, output_data = data
    if type == 'NDA_simple':
        I, O = decode(input_data, plateau_data), decode(input_data, selected_plateau_data)
        batch_size, seq_len, _ = output_data.shape
        _, n_sat, dim = O.shape

        O_reshaped = O.reshape(batch_size, seq_len, n_sat, dim)
        M = O_reshaped.mean(dim=1, keepdim=True)
        Std = torch.norm(O_reshaped - M, dim=[1, 2, 3], p=2, keepdim=True) / np.sqrt((seq_len - 1) * n_sat * dim)
        Std = Std.expand(batch_size, seq_len, 1, 1).reshape(batch_size * seq_len, 1, 1)

        return I, O, Std
    elif type == 'NDA':
        return input_data, selected_plateau_data
    elif type == 'tracking':
        return input_data, selected_plateau_data, output_data, [add_mask, mult_mask], output_data.std(dim=[-1, -2], keepdim=True)
    elif type == 'complete':
        return input_data, output_data, [add_mask, mult_mask], output_data.std(dim=[-1, -2], keepdim=True)
    else:
        return ValueError('invalid type argument')

def decode(input_data, encode_data):
    n_vector = encode_data.shape[-2]
    dim = input_data.shape[-1]

    input_data_extended = torch.nn.functional.pad(input_data, (0, 0, 0, 1)).unsqueeze(-2).expand(-1, -1, n_vector, -1)

    shifted = torch.nn.functional.pad(torch.arange(0, encode_data.shape[1]).unsqueeze(1), (0, 1)).unsqueeze(0).unsqueeze(2) + encode_data
    indices = (shifted[..., 0] * (1 + shifted[..., -1]) - input_data.shape[1] * shifted[..., -1]).to(torch.int64).unsqueeze(-1).expand(-1, -1, -1, dim)

    decoded_data = torch.gather(input_data_extended, dim=1, index=indices).reshape(-1, n_vector, dim)

    return decoded_data

if __name__ == '__main__':
    T = MakeData(10, 6, 5, 500, 700, 50000, 0.1, bias='freq', distrib='log')
    T= MakeDataParallel(10, 6, 5, 500, 700, 50000, 0.1, bias='freq', distrib='log', max_inflight=1000)