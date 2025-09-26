import torch
from Tools.XMLTools import loadXmlAsObj, saveObjAsXml
import os
from tqdm import tqdm
import numpy as np
import time
from Inter.Model.Scenario import Simulator as GlobalSimulator, FreqBiasedSimulatorTemplate, BiasedSimulatorTemplate

class Simulator(GlobalSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{k: v for (k,v) in kwargs.items() if k != 'period_mes'})
        self.period_mesureur = kwargs['period_mes']
        self.mem_mesureur = []
        self.len_mem_mesureur = []

    def run(self):
        i = 0
        while self.sensor_simulator.running:
            if not i % self.period_mesureur:
                P = self.sensor_simulator.P
                TI = self.sensor_simulator.TI
                TM = self.sensor_simulator.TM
                t =  self.sensor_simulator.T

                P = [p + [t - tm, tm - ti] for p, ti, tm in zip(P, TI, TM)]

                self.len_mem_mesureur.append(len(P))
                if P == []:
                    P = torch.zeros((self.n_mes, self.dim + 2))
                else:
                    P = torch.tensor(P)
                    P = torch.nn.functional.pad(P, (0, 0, 0, self.n_mes - len(P)))

                self.mem_mesureur.append(P.tolist())
            i += self.step()
        self.mem_mesureur.append(torch.zeros((self.n_mes, self.dim + 2)).tolist())
        self.len_mem_mesureur.append(0)


class BiasedSimulator(BiasedSimulatorTemplate, Simulator):
    pass

class FreqBiasedSimulator(FreqBiasedSimulatorTemplate, Simulator):
    pass

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
        d_in, n_pulse_plateau, n_sat, n_mes, period_mes, len_in, len_out, sensitivity, weight_f, weight_l = args[1:]
        S = Simulator(n_pulse_plateau, len_in, d_in - 1, n_sat=n_sat, n_mes=n_mes, period_mes=period_mes,
                      sensitivity=sensitivity, WeightF=weight_f, WeightL=weight_l)
    elif args[0] == 'freq':
        std, mean, d_in, n_pulse_plateau, n_sat, n_mes, period_mes, len_in, len_out, sensitivity, weight_f, weight_l = args[1:]
        S = FreqBiasedSimulator(std, mean, n_pulse_plateau, len_in, d_in - 1, n_sat=n_sat, n_mes=n_mes, period_mes=period_mes,
                                sensitivity=sensitivity, WeightF=weight_f, WeightL=weight_l)
    elif args[0] == 'all':
        std, mean, d_in, n_pulse_plateau, n_sat, n_mes, period_mes, len_in, len_out, sensitivity, weight_f, weight_l = args[1:]
        S = BiasedSimulator(std, mean, n_pulse_plateau, len_in, d_in - 1, n_sat=n_sat, n_mes=n_mes, period_mes=period_mes,
                            sensitivity=sensitivity, WeightF=weight_f, WeightL=weight_l)
    else:
        raise ValueError

    S.run()

    input_seq = S.L
    len_output_seq = len(S.sensor_simulator.R)
    output_seq = (S.sensor_simulator.R + [[0.] * (d_in + 1)] * (len_out - len_output_seq))[:len_out]
    mem_mesureur = S.mem_mesureur
    len_mem_mesureur = S.len_mem_mesureur

    return input_seq, len_output_seq, output_seq, mem_mesureur, len_mem_mesureur

def MakeData(d_in, n_pulse_plateau, n_sat, n_mes, period_mes, len_in, len_out, n_data, sensitivity=0.1, weight_f=None, weight_l=None,
             bias='none', std_min=1, std_max=5, mean_min=-10, mean_max=10, distrib='log', plot=False):
    input_data, len_output_data, output_data, mem_data, len_mem_data = [], [], [], [], []
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
            input_seq, len_output_seq, output_seq, mem_mesureur, len_mem_mesureur = (
                generate_sample((bias, std, mean, d_in, n_pulse_plateau, n_sat, n_mes, period_mes, len_in, len_out, sensitivity, weight_f, weight_l)))
        else:
            input_seq, len_output_seq, output_seq, mem_mesureur, len_mem_mesureur = (
                generate_sample((bias, d_in, n_pulse_plateau, n_sat, n_mes, period_mes, len_in, len_out, sensitivity, weight_f, weight_l)))

        input_data.append(input_seq)
        len_output_data.append(len_output_seq)
        output_data.append(output_seq)
        mem_data.append(mem_mesureur)
        len_mem_data.append(len_mem_mesureur)

    input_data = torch.tensor(input_data, dtype=torch.float)
    output_data = torch.tensor(output_data, dtype=torch.float)
    mem_data = torch.tensor(mem_data, dtype=torch.float)
    len_mem_data = torch.tensor(len_mem_data, dtype=torch.float)

    len_element_output = torch.tensor(len_output_data).unsqueeze(-1)
    arrange = torch.arange(len_out + 1).unsqueeze(0).expand(n_data, -1)
    add_mask = torch.tensor(len_element_output == arrange, dtype=torch.float).unsqueeze(-1)
    mult_mask = torch.tensor(len_element_output >= arrange, dtype=torch.float).unsqueeze(-1)

    return input_data, add_mask, mult_mask, output_data, mem_data, len_mem_data

from concurrent.futures import ProcessPoolExecutor
from collections import deque

def MakeDataParallel(d_in, n_pulse_plateau, n_sat, n_mes, period_mes, len_in, len_out, n_data, sensitivity=0.1, weight_f=None, weight_l=None,
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
        args = [(bias, std_list[i], mean_list[i], d_in, n_pulse_plateau, n_sat, n_mes, period_mes, len_in, len_out, sensitivity, weight_f, weight_l) for i in range(n_data)]

    else:
        args = [(bias, d_in, n_pulse_plateau, n_sat, n_mes, period_mes, len_in, len_out, sensitivity, weight_f, weight_l) for _ in range(n_data)]

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
    input_data, len_output_data, output_data, mem_data, len_mem_data = zip(*results)

    input_data = torch.tensor(input_data, dtype=torch.float)
    output_data = torch.tensor(output_data, dtype=torch.float)
    mem_data = torch.tensor(mem_data, dtype=torch.float)
    len_mem_data = torch.tensor(len_mem_data, dtype=torch.float)

    len_element_output = torch.tensor(len_output_data).unsqueeze(-1)
    arrange = torch.arange(len_out + 1).unsqueeze(0).expand(n_data, -1)
    add_mask = torch.tensor(len_element_output == arrange, dtype=torch.float).unsqueeze(-1)
    mult_mask = torch.tensor(len_element_output >= arrange, dtype=torch.float).unsqueeze(-1)

    return input_data, add_mask, mult_mask, output_data, mem_data, len_mem_data

def GetDataSecond(d_in, n_pulse_plateau, n_sat, n_mes, period_mes, len_in, len_out, n_data_training, n_data_validation=1000, sensitivity=0.1,
            weight_f=None, weight_l=None, bias='none', std_min=1., std_max=5., mean_min=-10., mean_max=10.,
            distrib='log', plot=False, save_path=None, parallel=False, max_inflight=None, executor=None):
    make_data = MakeDataParallel if parallel else MakeData

    if bias == 'none':
        kwargs = {'d_in': d_in,
                  'n_pulse_plateau': n_pulse_plateau,
                  'n_sat': n_sat,
                  'n_mes': n_mes,
                  'period_mes': period_mes,
                  'len_in': len_in,
                  'len_out': len_out,
                  'sensitivity': sensitivity}
    else:
        kwargs = {'d_in': d_in,
                  'n_pulse_plateau': n_pulse_plateau,
                  'n_sat': n_sat,
                  'n_mes': n_mes,
                  'period_mes': period_mes,
                  'len_in': len_in,
                  'len_out': len_out,
                  'sensitivity': sensitivity,
                  'bias': bias,
                  'std_min': std_min,
                  'std_max': std_max,
                  'mean_min': mean_min,
                  'mean_max': mean_max,
                  'distrib': distrib}

    if max_inflight is not None and parallel:
        kwargs['max_inflight'] = max_inflight

    if executor is not None:
        kwargs['executor'] = executor

    if plot or (save_path is None):
        weight_f = weight_f if weight_f is not None else np.array([1., 0.] + [0.] * (d_in - 1))
        weight_f = weight_f / np.linalg.norm(weight_f)
        weight_l = weight_l if weight_l is not None else np.array([0., 1.] + [0.] * (d_in - 1))
        weight_l = weight_l / np.linalg.norm(weight_l)

        if plot:
            data = make_data(n_data=n_data_training, weight_f=weight_f, weight_l=weight_l, plot=True, **kwargs)
            return return_data(*data)

        data_training = make_data(n_data=n_data_training, weight_f=weight_f, weight_l=weight_l, **kwargs)
        data_validation = make_data(n_data=n_data_validation, weight_f=weight_f, weight_l=weight_l, **kwargs)
        return [return_data(*data_training), return_data(*data_validation)]

    try:
        os.mkdir(save_path)
    except:
        pass

    for file in os.listdir(save_path):
        try:
            kwargs_file = loadXmlAsObj(os.path.join(save_path, file, 'kwargs.xml'))
        except:
            continue

        if kwargs_file == {k: v for k, v in kwargs.items() if not k in ['executor', 'max_inflight']}:
            weight_l = np.load(os.path.join(save_path, file, 'weight_l.npy'))
            weight_f = np.load(os.path.join(save_path, file, 'weight_f.npy'))

            n_data = {'training': n_data_training, 'validation': n_data_validation}
            output = []
            for phase in  n_data.keys():

                input_data_ = torch.load(os.path.join(save_path, file, phase + '_input_data'))
                add_mask_ = torch.load(os.path.join(save_path, file, phase + '_add_mask'))
                mult_mask_ = torch.load(os.path.join(save_path, file, phase + '_mult_mask'))
                output_data_ = torch.load(os.path.join(save_path, file, phase + '_output_data'))
                mem_data_ = torch.load(os.path.join(save_path, file, phase + '_mem_data'))
                len_mem_data_ = torch.load(os.path.join(save_path, file, phase + '_len_mem_data'))

                if len(input_data_) < n_data[phase]:
                    input_data, add_mask, mult_mask, output_data, mem_data, len_mem_data = (
                        make_data(n_data=n_data[phase] - len(input_data_), weight_f=weight_f, weight_l=weight_l, **kwargs))

                    input_data_ = torch.cat((input_data_, input_data), dim=0)
                    add_mask_ = torch.cat((add_mask_, add_mask), dim=0)
                    mult_mask_ = torch.cat((mult_mask_, mult_mask), dim=0)
                    output_data_ = torch.cat((output_data_, output_data), dim=0)
                    mem_data_ = torch.cat((mem_data_, mem_data), dim=0)
                    len_mem_data_ = torch.cat((len_mem_data_, len_mem_data), dim=0)

                    torch.save(input_data_, os.path.join(save_path, file, phase + '_input_data'))
                    torch.save(add_mask_, os.path.join(save_path, file, phase + '_add_mask'))
                    torch.save(mult_mask_, os.path.join(save_path, file, phase + '_mult_mask'))
                    torch.save(output_data_, os.path.join(save_path, file, phase + '_output_data'))
                    torch.save(mem_data_, os.path.join(save_path, file, phase + '_mem_data'))
                    torch.save(len_mem_data_, os.path.join(save_path, file, phase + '_len_mem_data'))

                output.append(list(return_data(input_data_[:n_data[phase]],
                                               add_mask_[:n_data[phase]],
                                               mult_mask_[:n_data[phase]],
                                               output_data_[:n_data[phase]],
                                               mem_data_[:n_data[phase]],
                                               len_mem_data_[:n_data[phase]]
                                               )))

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

    weight_f = weight_f if weight_f is not None else np.array([1., 0.] + [0.] * (d_in - 1))
    weight_f = weight_f / np.linalg.norm(weight_f)
    weight_l = weight_l if weight_l is not None else np.array([0., 1.] + [0.] * (d_in - 1))
    weight_l = weight_l / np.linalg.norm(weight_l)
    np.save(os.path.join(save_path, file, 'weight_f'), weight_f)
    np.save(os.path.join(save_path, file, 'weight_l'), weight_l)
    saveObjAsXml({k: v for k, v in kwargs.items() if not k in ['executor', 'max_inflight']}, os.path.join(save_path, file, 'kwargs.xml'))

    n_data = {'training': n_data_training, 'validation': n_data_validation}
    output = []
    for phase in n_data.keys():

        input_data, add_mask, mult_mask, output_data, mem_data, len_mem_data = (
            make_data(n_data=n_data[phase], weight_f=weight_f, weight_l=weight_l, **kwargs))

        torch.save(input_data, os.path.join(save_path, file, phase + '_input_data'))
        torch.save(add_mask, os.path.join(save_path, file, phase + '_add_mask'))
        torch.save(mult_mask, os.path.join(save_path, file, phase + '_mult_mask'))
        torch.save(output_data, os.path.join(save_path, file, phase + '_output_data'))
        torch.save(mem_data, os.path.join(save_path, file, phase + '_mem_data'))
        torch.save(len_mem_data, os.path.join(save_path, file, phase + '_len_mem_data'))

        output.append(list(return_data(input_data,
                                       add_mask,
                                       mult_mask,
                                       output_data,
                                       mem_data,
                                       len_mem_data)))

    return output

def return_data(*args):
    input_data, add_mask, mult_mask, output_data, mem_data, len_mem_data = args
    Mask = mult_mask[:, :-1]
    M = output_data.sum(dim=1, keepdim=True).expand(output_data.shape) / (Mask.sum(dim=1, keepdim=True) + 1e-5) * Mask
    Std = torch.norm(M-output_data, dim=[1, 2], keepdim=True) / torch.sqrt(output_data.shape[2] * abs(torch.sum(Mask, dim=1, keepdim=True) - 1) + 1e-5)
    return input_data, output_data, mem_data, len_mem_data, [add_mask, mult_mask], Std


def GetData(d_in, n_pulse_plateau, n_sat, n_mes, len_in, len_out, n_data_training, n_data_validation=1000, sensitivity=0.1,
            weight_f=None, weight_l=None, bias='none', std_min=1., std_max=5., mean_min=-10., mean_max=10.,
            distrib='log', plot=False, save_path=None, parallel=False, size_tampon_source=10,
            size_focus_source=20, size_tampon_target=15, size_focus_target=30, max_inflight=None):
    window_param = {
        'size_tampon_source': size_tampon_source,
        'size_focus_source': size_focus_source,
        'size_tampon_target': size_tampon_target,
        'size_focus_target': size_focus_target,
    }

    if parallel:
        with ProcessPoolExecutor() as executor:
            data = GetDataSecond(d_in, n_pulse_plateau, n_sat, n_mes, size_focus_source, len_in, len_out, n_data_training,
                                 n_data_validation=n_data_validation, sensitivity=sensitivity, weight_f=weight_f,
                                 weight_l=weight_l, bias=bias, std_min=std_min, std_max=std_max, mean_min=mean_min,
                                 mean_max=mean_max, distrib=distrib, plot=plot, save_path=save_path, parallel=parallel,
                                 max_inflight=max_inflight, executor=executor)

    else:
        data = GetDataSecond(d_in, n_pulse_plateau, n_sat, n_mes, size_focus_source, len_in, len_out, n_data_training,
                             n_data_validation=n_data_validation, sensitivity=sensitivity, weight_f=weight_f,
                             weight_l=weight_l, bias=bias, std_min=std_min, std_max=std_max, mean_min=mean_min,
                             mean_max=mean_max, distrib=distrib, plot=plot, save_path=save_path, parallel=parallel,
                             max_inflight=max_inflight)

    if plot:
        input_data, output_data, mem_data, len_mem_data, [add_mask, mult_mask], _ = data
        return window(input_data, output_data, mem_data, len_mem_data, mult_mask, add_mask, window_param)

    else:
        data_training, data_validation = data
        input_data_t, output_data_t, mem_data_t, len_mem_data_t, [add_mask_t, mult_mask_t], _ = data_training
        input_data_v, output_data_v, mem_data_v, len_mem_data_v, [add_mask_v, mult_mask_v], _ = data_validation
        return (window(input_data_t, output_data_t, mem_data_t, len_mem_data_t, mult_mask_t, add_mask_t, window_param),
                window(input_data_v, output_data_v, mem_data_v, len_mem_data_v, mult_mask_v, add_mask_v, window_param))

def window(input_data, output_data, mem_data, len_mem_data, mult_mask, add_mask, param):
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

    time_of_arrival_window = size_focus_source * torch.arange(1, windowed_input_data.shape[1] + 1) - 1
    time_of_arrival_window[time_of_arrival_window >= input_data.shape[1]] = output_data.shape[1] - 1
    time_of_emission = torch.arange(output_data.shape[1]) - output_data[:, :, -1] + output_data[:, :, -2]
    time_of_emission = time_of_emission * mult_mask[:, 1:, 0] + (1 - mult_mask[:, 1:, 0]) * output_data.shape[1]
    pulse_arrived = torch.nn.functional.pad(torch.lt(
        time_of_emission.unsqueeze(-1),
        time_of_arrival_window.unsqueeze(0).unsqueeze(0)
    ), (1, 0), value=False).to(torch.float32)

    upcoming_pulse = pulse_arrived[:, :, 1:] - pulse_arrived[:, :, :-1]

    idx = torch.arange(size_tampon_target + size_focus_target).view(1, 1, size_tampon_target + size_focus_target)
    starts = upcoming_pulse.argmax(dim=1).unsqueeze(-1)
    window_indices = starts + idx

    window_mask = torch.lt(
        torch.arange(-size_tampon_target, size_focus_target).view(1, 1, size_tampon_target + size_focus_target),
        upcoming_pulse.sum(dim=1).unsqueeze(-1)
    ).unsqueeze(-1).to(torch.float32)

    window_mask[:, :, :size_tampon_target] = 0

    # modification de l'encodage du ToA dans la séquence de sortie
    output_data[:, :, -1] = - output_data[:, :, -1] + torch.arange(output_data.shape[1]).view(1, -1) * mult_mask[:, 1:, 0]
    new_output_data = torch.nn.functional.pad(output_data, (0, 0, size_tampon_target, size_focus_target))

    mask = torch.zeros(*new_input_data.shape[:-1], 1)
    mask[:, :size_tampon_source] = 1
    mask[:, -size_focus_source:] = 1
    mask[:, -size_focus_source] = -1
    mask = mask.unfold(1, size_tampon_source + size_focus_source, size_focus_source).transpose(2, 3)
    mask = mask.reshape(-1, *mask.shape[2:])
    source_pad_mask = (mask + mask ** 2) / 2
    source_end_mask = (-mask + mask ** 2) / 2

    batch_size = len(new_output_data)
    batch_idx = torch.arange(batch_size).view(batch_size, 1, 1).expand(-1, windowed_input_data.shape[1], size_tampon_target + size_focus_target)
    windowed_output_data = new_output_data[batch_idx, window_indices]
    mask = torch.nn.functional.pad((1 - mult_mask - add_mask), (0, 0, size_tampon_target, size_focus_target - 1), value=1)
    mask = mask[batch_idx, window_indices]

    windowed_output_data[:, :, :, -1] = windowed_output_data[:, :, :, -1] - (
        torch.arange(windowed_output_data.shape[1]).view(1, -1, 1) * size_focus_source +
        torch.arange(-size_tampon_target, size_focus_target).view(1, 1, -1)
    )

    mask = mask.reshape(-1, *mask.shape[2:])
    target_end_mask = (-mask + mask ** 2) / 2
    target_pad_mask = (mask + mask ** 2) / 2

    mean = (windowed_output_data * window_mask).sum(dim=2, keepdim=True).expand(windowed_output_data.shape) / (
            window_mask.sum(dim=2, keepdim=True) + 1e-5) * window_mask

    std = torch.norm(mean - windowed_output_data * window_mask, dim=[1, 2, 3], p=2) / (
            windowed_output_data.shape[-1] * (window_mask.sum(dim=[1, 2, 3]) - windowed_output_data.shape[1])).sqrt()

    std_output = std.unsqueeze(-1).expand(-1, windowed_output_data.shape[1]).reshape(-1, 1, 1)

    windowed_output_data = windowed_output_data.reshape(-1, *windowed_output_data.shape[2:])
    windowed_input_data = windowed_input_data.reshape(-1, *windowed_input_data.shape[2:]).transpose(1, 2)
    window_mask = window_mask.reshape(-1, *window_mask.shape[2:]) + target_end_mask

    mem_in_data = mem_data[:, :-1].reshape(-1, *mem_data.shape[2:])
    mem_out_data = mem_data[:, 1:].reshape(-1, *mem_data.shape[2:])

    mem_mask_in = torch.arange(mem_data.shape[2]).unsqueeze(0).unsqueeze(1).unsqueeze(-1) >= len_mem_data[:, :-1].unsqueeze(2).unsqueeze(-1)
    mem_mask_out = torch.arange(mem_data.shape[2]).unsqueeze(0).unsqueeze(1).unsqueeze(-1) >= len_mem_data[:, 1:].unsqueeze(2).unsqueeze(-1)
    mem_mask_in = mem_mask_in.to(torch.float)
    mem_mask_out = mem_mask_out.to(torch.float)

    mean = (mem_data[:, 1:] * (1 - mem_mask_out)).sum(dim=2, keepdim=True).expand(mem_data[:, 1:].shape) / (
            (1 - mem_mask_out).sum(dim=2, keepdim=True) + 1e-5) * (1 - mem_mask_out)

    std = torch.norm(mean - mem_data[:, 1:] * (1 - mem_mask_out), dim=[1, 2, 3], p=2) / (
            mem_data[:, 1:].shape[-1] * ((1 - mem_mask_out).sum(dim=[1, 2, 3]) - mem_data[:, 1:].shape[1])).sqrt()

    std_mem = std.unsqueeze(-1).expand(-1, mem_data[:, 1:].shape[1]).reshape(-1, 1, 1)

    mem_mask_in = mem_mask_in.reshape(*mem_in_data.shape[:2], 1)
    mem_mask_out = mem_mask_out.reshape(*mem_in_data.shape[:2], 1)

    return (windowed_input_data, windowed_output_data, mem_in_data, mem_out_data,
            [source_pad_mask, source_end_mask, target_pad_mask, target_end_mask,
             mem_mask_in, mem_mask_out, window_mask], std_output, std_mem)

if __name__ == '__main__':
    T = GetData(9, 7, 5, 6, 100, 120, 1000, n_data_validation=1000, sensitivity=0.1,
            weight_f=None, weight_l=None, bias='none', std_min=1., std_max=5., mean_min=-10., mean_max=10.,
            distrib='log', plot=True, save_path=None, parallel=True, size_tampon_source=3,
            size_focus_source=7, size_tampon_target=4, size_focus_target=9, max_inflight=100)