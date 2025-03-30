import torch
from Inter.Model.Scenario import Simulator, BiasedSimulator, FreqBiasedSimulator
from Tools.XMLTools import loadXmlAsObj, saveObjAsXml
import os
from tqdm import tqdm
import numpy as np

def generate_sample(bias, *args):
    import warnings
    warnings.filterwarnings("ignore")  # Désactive les warnings

    print(bias)
    """ Génère un échantillon unique basé sur les paramètres. """
    if bias == 'none':
        d_in, n_pulse_plateau, len_in, len_out, sensitivity = args
        S = Simulator(n_pulse_plateau, len_in, d_in - 1, sensitivity=sensitivity)
    elif bias == 'freq':
        std, mean, d_in, n_pulse_plateau, len_in, len_out, sensitivity = args
        S = FreqBiasedSimulator(std, mean, n_pulse_plateau, len_in, d_in - 1, sensitivity=sensitivity)
    elif bias == 'all':
        std, mean, d_in, n_pulse_plateau, len_in, len_out, sensitivity = args
        S = BiasedSimulator(std, mean, n_pulse_plateau, len_in, d_in - 1, sensitivity=sensitivity)
    else:
        raise ValueError

    S.run()

    input_data = S.L
    output = S.sensor_simulator.R
    len_element_output = len(output)

    # Remplissage pour correspondre à len_out
    output += [[0.] * (d_in + 1)] * (len_out - len(output))

    return input_data, output[:len_out], len_element_output

def MakeData(d_in, n_pulse_plateau, len_in, len_out, n_data, sensitivity, bias='none',
             std_min=1, std_max=5, mean_min=-10, mean_max=10, distrib='log', plot=False):
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

    for i in tqdm(range(n_data)):
        if bias != 'none':
            mean = mean_list[i]
            std = std_list[i]
            input, output, len_output = generate_sample(bias, std, mean, d_in, n_pulse_plateau, len_in, len_out, sensitivity)
        else:
            input, output, len_output = generate_sample(bias, d_in, n_pulse_plateau, len_in, len_out, sensitivity)
        input_data.append(input)
        output_data.append(output)
        len_element_output.append(len_output)

    len_element_output = torch.tensor(len_element_output).unsqueeze(-1)
    arange = torch.arange(len_out + 1).unsqueeze(0).expand(n_data, -1)
    add_mask = torch.tensor((len_element_output + 1) == arange, dtype=torch.float).unsqueeze(-1)
    mult_mask = torch.tensor((len_element_output + 1) >= arange, dtype=torch.float).unsqueeze(-1)

    return torch.tensor(input_data, dtype=torch.float), torch.tensor(output_data, dtype=torch.float), [add_mask, mult_mask]

from concurrent.futures import ProcessPoolExecutor

def MakeDataParallel(d_in, n_pulse_plateau, len_in, len_out, n_data, sensitivity, bias='none',
                     std_min=1, std_max=5, mean_min=-10, mean_max=10, distrib='log', plot=False):

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

        """ Génère n_data échantillons en parallèle avec ProcessPoolExecutor. """
        args = [(bias, std_list[i], mean_list[i], d_in, n_pulse_plateau, len_in, len_out, sensitivity) for i in range(n_data)]

    else:
        args = [(bias, d_in, n_pulse_plateau, len_in, len_out, sensitivity) for _ in range(n_data)]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(generate_sample, args))  # On passe une fonction globale et une liste d'args

    # Extraction des résultats
    input_data, output_data, len_element_output = zip(*results)

    # Conversion en tensors
    input_data = torch.tensor(input_data, dtype=torch.float)
    output_data = torch.tensor(output_data, dtype=torch.float)
    len_element_output = torch.tensor(len_element_output).unsqueeze(-1)

    # Masques
    arange = torch.arange(len_out + 1).unsqueeze(0).expand(n_data, -1)
    add_mask = torch.tensor((len_element_output + 1) == arange, dtype=torch.float).unsqueeze(-1)
    mult_mask = torch.tensor((len_element_output + 1) >= arange, dtype=torch.float).unsqueeze(-1)

    return input_data, output_data, [add_mask, mult_mask]


def GetData(d_in, n_pulse_plateau, len_in, len_out, n_data_training, n_data_validation, sensitivity, save_path=None, parallel=False):
    make_data = MakeDataParallel if parallel else MakeData

    if save_path is None:
        return [make_data(d_in, n_pulse_plateau, len_in, len_out, n_data_validation, sensitivity),
                make_data(d_in, n_pulse_plateau, len_in, len_out, n_data_training, sensitivity)]

    else:
        arg = {'d_in': d_in,
               'n_pulse_plateau': n_pulse_plateau,
               'len_in': len_in,
               'len_out': len_out,
               'sensitivity': sensitivity}

        for file in os.listdir(save_path):
            arg_file = loadXmlAsObj(os.path.join(save_path, file, 'arg.xml'))
            if arg_file == arg:
                InputTraining = torch.load(os.path.join(save_path, file, 'InputTraining'))
                OutputTraining = torch.load(os.path.join(save_path, file, 'OutputTraining'))
                AddMaskTraining = torch.load(os.path.join(save_path, file, 'AddMaskTraining'))
                MultMaskTraining = torch.load(os.path.join(save_path, file, 'MultMaskTraining'))

                if len(InputTraining) < n_data_training:
                    Input, Output, Mask = make_data(d_in, n_pulse_plateau, len_in, len_out, n_data_training - len(InputTraining), sensitivity)
                    InputTraining = torch.cat((InputTraining, Input), dim=0)
                    OutputTraining = torch.cat((OutputTraining, Output), dim=0)
                    AddMaskTraining = torch.cat((AddMaskTraining, Mask[0]), dim=0)
                    MultMaskTraining = torch.cat((MultMaskTraining, Mask[1]), dim=0)
                    torch.save(InputTraining, os.path.join(save_path, file, 'InputTraining'))
                    torch.save(OutputTraining, os.path.join(save_path, file, 'OutputTraining'))
                    torch.save(AddMaskTraining, os.path.join(save_path, file, 'AddMaskTraining'))
                    torch.save(MultMaskTraining, os.path.join(save_path, file, 'MultMaskTraining'))

                InputValidation = torch.load(os.path.join(save_path, file, 'InputValidation'))
                OutputValidation = torch.load(os.path.join(save_path, file, 'OutputValidation'))
                AddMaskValidation = torch.load(os.path.join(save_path, file, 'AddMaskValidation'))
                MultMaskValidation = torch.load(os.path.join(save_path, file, 'MultMaskValidation'))

                if len(InputValidation) < n_data_validation:
                    Input, Output, Mask = make_data(d_in, n_pulse_plateau, len_in, len_out, n_data_validation - len(InputValidation), sensitivity)
                    InputValidation = torch.cat((InputValidation, Input), dim=0)
                    OutputValidation = torch.cat((OutputValidation, Output), dim=0)
                    AddMaskValidation = torch.cat((AddMaskValidation, Mask[0]), dim=0)
                    MultMaskValidation = torch.cat((MultMaskValidation, Mask[1]), dim=0)
                    torch.save(InputValidation, os.path.join(save_path, file, 'InputValidation'))
                    torch.save(OutputValidation, os.path.join(save_path, file, 'OutputValidation'))
                    torch.save(AddMaskValidation, os.path.join(save_path, file, 'AddMaskValidation'))
                    torch.save(MultMaskValidation, os.path.join(save_path, file, 'MultMaskValidation'))

                return [[InputValidation[:n_data_validation], OutputValidation[:n_data_validation],
                         [AddMaskValidation[:n_data_validation], MultMaskValidation[:n_data_validation]]],
                        [InputTraining[:n_data_training], OutputTraining[:n_data_training],
                         [AddMaskTraining[:n_data_training], MultMaskTraining[:n_data_training]]]]

        file = 'config' + str(len(os.listdir(save_path)))
        os.mkdir(os.path.join(save_path, file))
        saveObjAsXml(arg, os.path.join(save_path, file, 'arg.xml'))
        InputValidation, OutputValidation, MaskValidation = make_data(d_in, n_pulse_plateau, len_in, len_out, n_data_validation, sensitivity)
        AddMaskValidation, MultMaskValidation = MaskValidation
        torch.save(InputValidation, os.path.join(save_path, file, 'InputValidation'))
        torch.save(OutputValidation, os.path.join(save_path, file, 'OutputValidation'))
        torch.save(AddMaskValidation, os.path.join(save_path, file, 'AddMaskValidation'))
        torch.save(MultMaskValidation, os.path.join(save_path, file, 'MultMaskValidation'))
        InputTraining, OutputTraining, MaskTraining = make_data(d_in, n_pulse_plateau, len_in, len_out, n_data_training, sensitivity)
        AddMaskTraining, MultMaskTraining = MaskTraining
        torch.save(InputTraining, os.path.join(save_path, file, 'InputTraining'))
        torch.save(OutputTraining, os.path.join(save_path, file, 'OutputTraining'))
        torch.save(AddMaskTraining, os.path.join(save_path, file, 'AddMaskTraining'))
        torch.save(MultMaskTraining, os.path.join(save_path, file, 'MultMaskTraining'))

        return [[InputValidation, OutputValidation, MaskValidation],
                [InputTraining, OutputTraining, MaskTraining]]



if __name__ == '__main__':
    import time

    # t = time.time()
    # I, O, len_el_O = MakeData(10, 5, 300, 400, 100, 0.1, bias='freq')
    # print(time.time() - t)

    t = time.time()
    I, O, len_el_O = MakeDataParallel(10, 5, 300, 400, 100, 0.1, bias='freq')
    print(time.time() - t)
