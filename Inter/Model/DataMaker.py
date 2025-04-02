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
        d_in, n_pulse_plateau, len_in, len_out, sensitivity, weight_f, weight_l = args[2:]
        S = Simulator(n_pulse_plateau, len_in, d_in - 1,
                      sensitivity=sensitivity, WeightF=weight_f, WeightL=weight_l)
    elif args[0] == 'freq':
        std, mean, d_in, n_pulse_plateau, len_in, len_out, sensitivity, weight_f, weight_l = args[2:]
        S = FreqBiasedSimulator(std, mean, n_pulse_plateau, len_in, d_in - 1,
                                sensitivity=sensitivity, WeightF=weight_f, WeightL=weight_l)
    elif args[0] == 'all':
        std, mean, d_in, n_pulse_plateau, len_in, len_out, sensitivity, weight_f, weight_l = args[2:]
        S = BiasedSimulator(std, mean, n_pulse_plateau, len_in, d_in - 1,
                            sensitivity=sensitivity, WeightF=weight_f, WeightL=weight_l)
    else:
        raise ValueError

    S.run()

    if args[1] == 'complete':
        input = S.L
        output = S.sensor_simulator.R
        len_element_output = len(output)

        # Remplissage pour correspondre à len_out
        output += [[0.] * (d_in + 1)] * (len_out - len(output))

        return input, output[:len_out], len_element_output
    elif args[1] == 'NDA':
        input = S.L
        output = S.D

        return input, output

    elif args[1] == 'tracking':
        input = S.D
        output = S.sensor_simulator.R
        len_element_output = len(output)

        # Remplissage pour correspondre à len_out
        output += [[0.] * (d_in + 1)] * (len_out - len(output))

        return input, output[:len_out], len_element_output

    else:
        raise ValueError('invalid type')

def MakeData(d_in, n_pulse_plateau, len_in, len_out, n_data, sensitivity, type='complete', weight_f=None, weight_l=None,
             bias='none', std_min=1, std_max=5, mean_min=-10, mean_max=10, distrib='log', plot=False):
    input_data = []
    output_data = []
    len_element_output = []

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
            input, output, *len_output = generate_sample((bias, type, std, mean, d_in, n_pulse_plateau, len_in, len_out, sensitivity, weight_f, weight_l))
        else:
            input, output, *len_output = generate_sample((bias, type, d_in, n_pulse_plateau, len_in, len_out, sensitivity, weight_f, weight_l))
        input_data.append(input)
        output_data.append(output)
        len_element_output.append(len_output)

    input_data = torch.tensor(input_data, dtype=torch.float)
    output_data = torch.tensor(output_data, dtype=torch.float)
    if type == 'complete' or type == 'tracking':
        len_element_output = torch.tensor(len_element_output)
        arange = torch.arange(len_out + 1).unsqueeze(0).expand(n_data, -1)
        add_mask = torch.tensor((len_element_output + 1) == arange, dtype=torch.float).unsqueeze(-1)
        mult_mask = torch.tensor((len_element_output + 1) >= arange, dtype=torch.float).unsqueeze(-1)

        return (input_data,
                output_data,
                output_data.std(dim=[-1, -2], keepdim=True),
                [add_mask, mult_mask])

    elif type == 'NDA':
        return input_data, output_data, output_data.std(dim=[-1, -2], keepdim=True)

    else:
        raise ValueError('invalid type')


from concurrent.futures import ProcessPoolExecutor

def MakeDataParallel(d_in, n_pulse_plateau, len_in, len_out, n_data, sensitivity, type='complete', weight_f=None, weight_l=None,
                     bias='none', std_min=1, std_max=5, mean_min=-10, mean_max=10, distrib='log', plot=False):

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
        args = [(bias, type, std_list[i], mean_list[i], d_in, n_pulse_plateau, len_in, len_out, sensitivity, weight_f, weight_l) for i in range(n_data)]

    else:
        args = [(bias, type, d_in, n_pulse_plateau, len_in, len_out, sensitivity, weight_f, weight_l) for _ in range(n_data)]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(generate_sample, args))  # On passe une fonction globale et une liste d'args

    # Extraction des résultats
    input_data, output_data, *len_element_output = zip(*results)

    input_data = torch.tensor(input_data, dtype=torch.float)
    output_data = torch.tensor(output_data, dtype=torch.float)
    if type == 'complete' or type == 'tracking':
        len_element_output = torch.tensor(len_element_output[0]).unsqueeze(-1)
        arange = torch.arange(len_out + 1).unsqueeze(0).expand(n_data, -1)
        add_mask = torch.tensor((len_element_output + 1) == arange, dtype=torch.float).unsqueeze(-1)
        mult_mask = torch.tensor((len_element_output + 1) >= arange, dtype=torch.float).unsqueeze(-1)

        return (input_data,
                output_data,
                output_data.std(dim=[-1, -2], keepdim=True),
                [add_mask, mult_mask])

    elif type == 'NDA':
        return input_data, output_data, output_data.std(dim=[-1, -2], keepdim=True)

    else:
        raise ValueError('invalid type')

def GetData(d_in, n_pulse_plateau, len_in, len_out, n_data_training, n_data_validation, sensitivity,
            weight_f=None, weight_l=None, bias='none', std_min=1, std_max=5, mean_min=-10, mean_max=10,
            distrib='log', plot=False, save_path=None, parallel=False, type='complete'):
    make_data = MakeDataParallel if parallel else MakeData

    try:
        os.mkdir(save_path)
    except:
        pass

    if save_path is None:
        return [make_data(d_in, n_pulse_plateau, len_in, len_out, n_data_validation, sensitivity, weight_f, weight_l, type=type),
                make_data(d_in, n_pulse_plateau, len_in, len_out, n_data_training, sensitivity, weight_f, weight_l, type=type)]

    else:
        if bias == 'none':
            kwargs = {'d_in': d_in,
                      'n_pulse_plateau': n_pulse_plateau,
                      'len_in': len_in,
                      'len_out': len_out,
                      'sensitivity': sensitivity,
                      'type': type}
        else:
            kwargs = {'d_in': d_in,
                      'n_pulse_plateau': n_pulse_plateau,
                      'len_in': len_in,
                      'len_out': len_out,
                      'sensitivity': sensitivity,
                      'type': type,
                      'bias': bias,
                      'std_min': std_min,
                      'std_max': std_max,
                      'mean_min': mean_min,
                      'mean_max': mean_max,
                      'distrib': distrib,
                      'plot': plot}

        for file in os.listdir(save_path):
            try:
                kwargs_file = loadXmlAsObj(os.path.join(save_path, file, 'kwargs.xml'))
            except:
                continue

            if kwargs_file == kwargs:
                weight_l = np.load(os.path.join(save_path, file, 'weight_l.npy'))
                weight_f = np.load(os.path.join(save_path, file, 'weight_f.npy'))
                InputTraining = torch.load(os.path.join(save_path, file, 'InputTraining'))
                OutputTraining = torch.load(os.path.join(save_path, file, 'OutputTraining'))
                StdTraining = torch.load(os.path.join(save_path, file, 'StdTraining'))

                if type == 'complete' or type == 'tracking':
                    AddMaskTraining = torch.load(os.path.join(save_path, file, 'AddMaskTraining'))
                    MultMaskTraining = torch.load(os.path.join(save_path, file, 'MultMaskTraining'))

                if len(InputTraining) < n_data_training:
                    Input, Output, Std, *Mask  = make_data(n_data=n_data_training - len(InputTraining), weight_f=weight_f, weight_l=weight_l, **kwargs)
                    InputTraining = torch.cat((InputTraining, Input), dim=0)
                    OutputTraining = torch.cat((OutputTraining, Output), dim=0)
                    StdTraining = torch.cat((StdTraining, Std), dim=0)
                    torch.save(InputTraining, os.path.join(save_path, file, 'InputTraining'))
                    torch.save(OutputTraining, os.path.join(save_path, file, 'OutputTraining'))
                    torch.save(StdTraining, os.path.join(save_path, file, 'StdTraining'))

                    if type == 'complete' or type == 'tracking':
                        Mask = Mask[0]
                        AddMaskTraining = torch.cat((AddMaskTraining, Mask[0]), dim=0)
                        MultMaskTraining = torch.cat((MultMaskTraining, Mask[1]), dim=0)
                        torch.save(AddMaskTraining, os.path.join(save_path, file, 'AddMaskTraining'))
                        torch.save(MultMaskTraining, os.path.join(save_path, file, 'MultMaskTraining'))

                InputValidation = torch.load(os.path.join(save_path, file, 'InputValidation'))
                OutputValidation = torch.load(os.path.join(save_path, file, 'OutputValidation'))
                StdValidation = torch.load(os.path.join(save_path, file, 'StdValidation'))

                if type == 'complete' or type == 'tracking':
                    AddMaskValidation = torch.load(os.path.join(save_path, file, 'AddMaskValidation'))
                    MultMaskValidation = torch.load(os.path.join(save_path, file, 'MultMaskValidation'))

                if len(InputValidation) < n_data_validation:
                    Input, Output, Std, *Mask = make_data(n_data=n_data_validation - len(InputValidation), weight_f=weight_f, weight_l=weight_l, **kwargs)
                    InputValidation = torch.cat((InputValidation, Input), dim=0)
                    OutputValidation = torch.cat((OutputValidation, Output), dim=0)
                    StdValidation = torch.cat((StdValidation, Std), dim=0)
                    torch.save(InputValidation, os.path.join(save_path, file, 'InputValidation'))
                    torch.save(OutputValidation, os.path.join(save_path, file, 'OutputValidation'))
                    torch.save(StdValidation, os.path.join(save_path, file, 'StdValidation'))

                    if type == 'complete' or type == 'tracking':
                        Mask = Mask[0]
                        AddMaskValidation = torch.cat((AddMaskValidation, Mask[0]), dim=0)
                        MultMaskValidation = torch.cat((MultMaskValidation, Mask[1]), dim=0)
                        torch.save(AddMaskValidation, os.path.join(save_path, file, 'AddMaskValidation'))
                        torch.save(MultMaskValidation, os.path.join(save_path, file, 'MultMaskValidation'))

                if type == 'complete' or type == 'tracking':
                    return [[InputValidation[:n_data_validation], OutputValidation[:n_data_validation],
                             [AddMaskValidation[:n_data_validation], MultMaskValidation[:n_data_validation]],
                             StdValidation],
                            [InputTraining[:n_data_training], OutputTraining[:n_data_training],
                             [AddMaskTraining[:n_data_training], MultMaskTraining[:n_data_training]],
                             StdTraining]]

                elif type == 'NDA':
                    return [[InputValidation[:n_data_validation], OutputValidation[:n_data_validation],
                             StdValidation],
                            [InputTraining[:n_data_training], OutputTraining[:n_data_training],
                             StdTraining]]

                else:
                    raise ValueError('invalid type')

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
        weight_l = weight_f if weight_f is not None else np.array([0., 1.] + [0.] * (d_in - 3))
        weight_l = weight_l / np.linalg.norm(weight_l)
        np.save(os.path.join(save_path, file, 'weight_f'), weight_f)
        np.save(os.path.join(save_path, file, 'weight_l'), weight_l)
        saveObjAsXml(kwargs, os.path.join(save_path, file, 'kwargs.xml'))

        InputValidation, OutputValidation, StdValidation, *MaskValidation = make_data(n_data=n_data_validation, weight_f=weight_f, weight_l=weight_l, **kwargs)
        torch.save(InputValidation, os.path.join(save_path, file, 'InputValidation'))
        torch.save(OutputValidation, os.path.join(save_path, file, 'OutputValidation'))
        torch.save(StdValidation, os.path.join(save_path, file, 'StdValidation'))

        InputTraining, OutputTraining, StdTraining, *MaskTraining = make_data(n_data=n_data_training, weight_f=weight_f, weight_l=weight_l, **kwargs)
        torch.save(InputTraining, os.path.join(save_path, file, 'InputTraining'))
        torch.save(OutputTraining, os.path.join(save_path, file, 'OutputTraining'))
        torch.save(StdTraining, os.path.join(save_path, file, 'StdTraining'))

        if type == 'complete' or type == 'tracking':
            AddMaskValidation, MultMaskValidation = MaskValidation[0]
            torch.save(AddMaskValidation, os.path.join(save_path, file, 'AddMaskValidation'))
            torch.save(MultMaskValidation, os.path.join(save_path, file, 'MultMaskValidation'))

            AddMaskTraining, MultMaskTraining = MaskTraining[0]
            torch.save(AddMaskTraining, os.path.join(save_path, file, 'AddMaskTraining'))
            torch.save(MultMaskTraining, os.path.join(save_path, file, 'MultMaskTraining'))

            return [[InputValidation, OutputValidation, [AddMaskValidation, MultMaskValidation], StdValidation],
                    [InputTraining, OutputTraining, [AddMaskTraining, MultMaskTraining], StdTraining]]

        elif type == 'NDA':
            return [[InputValidation, OutputValidation, StdValidation],
                    [InputTraining, OutputTraining, StdTraining]]

        else:
            raise ValueError('invalid type')



if __name__ == '__main__':
    import time

    t = time.time()
    I, O, Std, *masks = MakeData(10, 5, 30, 40, 100, 0.1, bias='freq', type='NDA')
    print(time.time() - t)

    print('next')

    t = time.time()
    I, O, Std, *masks = MakeDataParallel(10, 5, 30, 40, 100, 0.1, bias='freq', type='complete')
    print(time.time() - t)
    #
    # t = time.time()
    # GetData(10, 5, 30, 40, 20, 10, 0.1,
    #         bias='none', std_min=1, std_max=5, mean_min=-10, mean_max=10, distrib='log', plot=False,
    #         save_path=r'C:\Users\Matth\Documents\Projets\Inter\Data', parallel=False)
    # print(time.time() - t)
    #
    # t = time.time()
    # GetData(10, 5, 31, 40, 2000, 500, 0.1,
    #         bias='none', std_min=1, std_max=5, mean_min=-10, mean_max=10, distrib='log', plot=False,
    #         save_path=r'C:\Users\Matth\Documents\Projets\Inter\Data', parallel=True)
    # print(time.time() - t)