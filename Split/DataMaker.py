import torch
import numpy as np
import os
from Tools.XMLTools import loadXmlAsObj, saveObjAsXml
from FakeDigitalTwin.Simulator import DigitalTwin
from FakeDigitalTwin.Pulse import Pulse
from tqdm import tqdm

def MakeData(arg_simulateur, len_in_full, len_in_window, len_out, len_out_temp, n_data, seed=None):
    if seed is not None:
        np.random.seed(seed)
    input_data = []
    len_element_input = []
    output_data = []
    len_element_output = []
    len_start_output = []

    # 1.06 correspond à la durée d'impulsion moyenne en unité de temps
    # 4 correspond à la densité moyenne du scénario
    # t_max est le temps maximal d'arrivé d'une impulsion
    t_max = len_in_full * 1.06 / 4
    # le facteur 0.5 permet d'avoir un glissement sur les impulsions d'entrée entre 2 fenêtres
    t_win = 0.5 * len_in_window * 1.06 / 4

    while len(input_data) < n_data:

        # On se donne un scénario de TMax unités de temps
        # On a donc en moyenne "density" impulsions en même temps
        TOA = t_max * np.sort(np.random.random(size=len_in_full))

        ####################################################################################################################
        # Pour un radar pulsé, si l'unité de temps est la microseconde, la durée de l'impulsion est entre 0.1 et 1
        # On ajoute aussi 3% d'impulsions en onde continue, de LI comprise entre 15 et 20
        # La durée d'impulsion moyenne est d'environ 1.06 unité de temps
        ####################################################################################################################
        LIcourte = np.random.uniform(0.1, 1, int(len_in_full * 0.97))
        LILongue = np.random.uniform(15, 20, len_in_full - int(len_in_full * 0.97))
        LI = np.concatenate([LILongue, LIcourte])
        np.random.shuffle(LI)

        ####################################################################################################################
        # Le niveau de saturation est de 10 unités, on veut que 98% des impulsions soit moins fortes

        # import numpy as np
        # Lvl_max = 10
        # scale = 3
        # frac = 1
        # while frac > 0.02:
        #     scale *= 0.999
        #     np_gamma = np.random.gamma(shape=2, scale=scale, size=100000)
        #     frac = sum(np_gamma > Lvl_max)/100000
        #     print(scale)
        #     print(frac)
        #     print('\n')
        #
        # scale = 1.725
        ####################################################################################################################
        Level = np.random.gamma(shape=2, scale=1.725, size=len_in_full)

        # Les fréquences se trouvent entre 0 et 10 et peuvent varier le long d'une impulsion de 0.05 unité de fréquence
        dF = 0.05 * (2 * np.random.random(size=len_in_full) - 1)
        FreqMoy = 9 * np.random.random(size=len_in_full) + 0.5
        FreqStart = FreqMoy + dF
        FreqEnd = FreqMoy - dF

        AntP = [Pulse(TOA=round(TOA[k], 3), LI=round(LI[k], 3), Level=round(Level[k], 3), FreqStart=round(FreqStart[k], 3),
                      FreqEnd=round(FreqEnd[k], 3), Id=k) for k in range(len_in_full)]

        DT = DigitalTwin(Param=arg_simulateur)

        DT.forward(AntP)

        input = [[pulse.TOA, pulse.LI, pulse.Level, pulse.FreqStart, pulse.FreqEnd] for pulse in AntP]
        output = [[pulse['TOA'], pulse['LI'], pulse['Level'], pulse['FreqMin'], pulse['FreqMax']] for pulse in DT.PDWs]
        output_Tp = DT.Tplist

        i = 1
        input_win = []
        output_win = [[0.] * 5] * len_out_temp
        nb_temp = len_out_temp

        while (input or output) and (len(input_data) < n_data):

            while input and (input[0][0] < t_win * i):
                input_win.append(input.pop(0))
                if len(input_win) > len_in_window:
                    input_win.pop(0)

            len_element_input.append(len(input_win))
            input_data.append([[0.] * 5] * (len_in_window - len(input_win)) + input_win)

            output_win = output_win[-len_out_temp:]
            len_start_output.append(max(nb_temp, 0))

            while output and (output_Tp[0] < t_win * i):
                output_Tp.pop(0)
                output_win.append(output.pop(0))
                nb_temp -= 1
                if len(output_win) > len_out:
                    output_win.pop(0)

            len_element_output.append(len(output_win))
            output_data.append(output_win + (output + [[0.] * 5] * len_out)[:(len_out - len(output_win))])

            i += 1

        print(str(100 * len(input_data) / n_data) + ' %', end='\r')

    len_element_output = torch.tensor(len_element_output).unsqueeze(-1)
    arange = torch.arange(len_out).unsqueeze(0).expand(len(output_data), -1)
    add_mask_output_end = torch.tensor((len_element_output + 1) == arange, dtype=torch.float).unsqueeze(-1)
    mult_mask_output_end = torch.tensor((len_element_output + 1) >= arange, dtype=torch.float).unsqueeze(-1)

    len_start_output = torch.tensor(len_start_output).unsqueeze(-1)
    mult_mask_output_start = torch.tensor(len_start_output > arange, dtype=torch.float).unsqueeze(-1)

    len_element_input = torch.tensor(len_element_input).unsqueeze(-1)
    arange = torch.arange(len_in_window).unsqueeze(0).expand(len(input_data), -1)
    mult_mask_input_start = torch.tensor((len_element_input + 1) <= arange.flip(-1), dtype=torch.float).unsqueeze(-1)

    return torch.tensor(input_data, dtype=torch.float), torch.tensor(output_data, dtype=torch.float), \
        [add_mask_output_end, mult_mask_output_end, mult_mask_output_start, mult_mask_input_start]

def GetData(len_in_full, len_in_window, len_out, len_out_temp, n_data_training, n_data_validation, save_path=None,
            Fe_List=[5.1, 5, 4.9, 4.8],
            Duree_max_impulsion=4,
            Seuil_mono=10,
            Seuil_harmo=8,
            Seuil_IM=8,
            Seuil_sensi_traitement=6,
            Seuil_sensi=1,
            Contraste_geneur=0.2,
            Nint=500,
            Contraste_geneur_2=1,
            M1_aveugle=2,
            M2_aveugle=2,
            M_local=5,
            N_DetEl=12,
            Seuil_ecart_freq=5e-3,
            Duree_maintien_max=0.2,
            N_mesureurs_max=8):

    arg_simulateur = {
        'Fe_List': Fe_List,
        'Duree_max_impulsion': Duree_max_impulsion,
        'Seuil_mono': Seuil_mono,
        'Seuil_harmo': Seuil_harmo,
        'Seuil_IM': Seuil_IM,
        'Seuil_sensi_traitement': Seuil_sensi_traitement,
        'Seuil_sensi': Seuil_sensi,
        'Contraste_geneur': Contraste_geneur,
        'Nint': Nint,
        'Contraste_geneur_2': Contraste_geneur_2,
        'M1_aveugle': M1_aveugle,
        'M2_aveugle': M2_aveugle,
        'M_local': M_local,
        'N_DetEl': N_DetEl,
        'Seuil_ecart_freq': Seuil_ecart_freq,
        'Duree_maintien_max': Duree_maintien_max,
        'N_mesureurs_max': N_mesureurs_max,
        'PDW_tries': 1,
    }

    arg = {'len_in_full': len_in_full,
           'len_in_window': len_in_window,
           'len_out_temp': len_out_temp,
           'len_out': len_out}
    arg.update(arg_simulateur)

    if save_path is None:
        return [MakeData(arg_simulateur, arg['len_in_full'], arg['len_in_window'], arg['len_out'], arg['len_out_temp'], n_data_validation),
                MakeData(arg_simulateur, arg['len_in_full'], arg['len_in_window'], arg['len_out'], arg['len_out_temp'], n_data_training)]

    else:
        for file in os.listdir(save_path):
            arg_file = loadXmlAsObj(os.path.join(save_path, file, 'arg.xml'))
            if arg_file == arg:
                InputTraining = torch.load(os.path.join(save_path, file, 'InputTraining'))
                OutputTraining = torch.load(os.path.join(save_path, file, 'OutputTraining'))
                AddMaskOutEndTraining = torch.load(os.path.join(save_path, file, 'AddMaskOutEndTraining'))
                MultMaskOutEndTraining = torch.load(os.path.join(save_path, file, 'MultMaskOutEndTraining'))
                MultMaskOutStartTraining = torch.load(os.path.join(save_path, file, 'MultMaskOutStartTraining'))
                MultMaskInStartTraining = torch.load(os.path.join(save_path, file, 'MultMaskInStartTraining'))

                if len(InputTraining) < n_data_training:
                    Input, Output, Mask = MakeData(arg_simulateur, arg['len_in_full'], arg['len_in_window'],
                                                   arg['len_out'], arg['len_out_temp'], n_data_training - len(InputTraining))
                    InputTraining = torch.cat((InputTraining, Input), dim=0)
                    OutputTraining = torch.cat((OutputTraining, Output), dim=0)
                    AddMaskOutEndTraining = torch.cat((AddMaskOutEndTraining, Mask[0]), dim=0)
                    MultMaskOutEndTraining = torch.cat((MultMaskOutEndTraining, Mask[1]), dim=0)
                    MultMaskOutStartTraining = torch.cat((MultMaskOutStartTraining, Mask[2]), dim=0)
                    MultMaskInStartTraining = torch.cat((MultMaskInStartTraining, Mask[3]), dim=0)
                    torch.save(InputTraining, os.path.join(save_path, file, 'InputTraining'))
                    torch.save(OutputTraining, os.path.join(save_path, file, 'OutputTraining'))
                    torch.save(AddMaskOutEndTraining, os.path.join(save_path, file, 'AddMaskOutEndTraining'))
                    torch.save(MultMaskOutEndTraining, os.path.join(save_path, file, 'MultMaskOutEndTraining'))
                    torch.save(MultMaskOutStartTraining, os.path.join(save_path, file, 'MultMaskOutStartTraining'))
                    torch.save(MultMaskInStartTraining, os.path.join(save_path, file, 'MultMaskInStartTraining'))

                InputValidation = torch.load(os.path.join(save_path, file, 'InputValidation'))
                OutputValidation = torch.load(os.path.join(save_path, file, 'OutputValidation'))
                AddMaskOutEndValidation = torch.load(os.path.join(save_path, file, 'AddMaskOutEndValidation'))
                MultMaskOutEndValidation = torch.load(os.path.join(save_path, file, 'MultMaskOutEndValidation'))
                MultMaskOutStartValidation = torch.load(os.path.join(save_path, file, 'MultMaskOutStartValidation'))
                MultMaskInStartValidation = torch.load(os.path.join(save_path, file, 'MultMaskInStartValidation'))

                if len(InputValidation) < n_data_validation:
                    Input, Output, Mask = MakeData(arg_simulateur, arg['len_in_full'], arg['len_in_window'],
                                                   arg['len_out'], arg['len_out_temp'], n_data_validation - len(InputValidation))
                    InputValidation = torch.cat((InputValidation, Input), dim=0)
                    OutputValidation = torch.cat((OutputValidation, Output), dim=0)
                    AddMaskOutEndValidation = torch.cat((AddMaskOutEndValidation, Mask[0]), dim=0)
                    MultMaskOutEndValidation = torch.cat((MultMaskOutEndValidation, Mask[1]), dim=0)
                    MultMaskOutStartValidation = torch.cat((MultMaskOutStartValidation, Mask[2]), dim=0)
                    MultMaskInStartValidation = torch.cat((MultMaskInStartValidation, Mask[3]), dim=0)
                    torch.save(InputValidation, os.path.join(save_path, file, 'InputValidation'))
                    torch.save(OutputValidation, os.path.join(save_path, file, 'OutputValidation'))
                    torch.save(AddMaskOutEndValidation, os.path.join(save_path, file, 'AddMaskOutEndValidation'))
                    torch.save(MultMaskOutEndValidation, os.path.join(save_path, file, 'MultMaskOutEndValidation'))
                    torch.save(MultMaskOutStartValidation, os.path.join(save_path, file, 'MultMaskOutStartValidation'))
                    torch.save(MultMaskInStartValidation, os.path.join(save_path, file, 'MultMaskInStartValidation'))


                return [[InputValidation[:n_data_validation], OutputValidation[:n_data_validation],
                         [AddMaskOutEndValidation[:n_data_validation], MultMaskOutEndValidation[:n_data_validation],
                          MultMaskOutStartValidation[:n_data_validation], MultMaskInStartValidation[:n_data_validation]]],
                        [InputTraining[:n_data_training], OutputTraining[:n_data_training],
                         [AddMaskOutEndTraining[:n_data_training], MultMaskOutEndTraining[:n_data_training],
                          MultMaskOutStartTraining[:n_data_training], MultMaskInStartTraining[:n_data_training]]]]

        file = 'config' + str(len(os.listdir(save_path)))
        os.mkdir(os.path.join(save_path, file))
        saveObjAsXml(arg, os.path.join(save_path, file, 'arg.xml'))
        InputValidation, OutputValidation, MaskValidation = MakeData(arg_simulateur, arg['len_in_full'], arg['len_in_window'],
                                                                     arg['len_out'], arg['len_out_temp'], n_data_validation)
        AddMaskOutEndValidation, MultMaskOutEndValidation, MultMaskOutStartValidation, MultMaskInStartValidation = MaskValidation
        torch.save(InputValidation, os.path.join(save_path, file, 'InputValidation'))
        torch.save(OutputValidation, os.path.join(save_path, file, 'OutputValidation'))
        torch.save(AddMaskOutEndValidation, os.path.join(save_path, file, 'AddMaskOutEndValidation'))
        torch.save(MultMaskOutEndValidation, os.path.join(save_path, file, 'MultMaskOutEndValidation'))
        torch.save(MultMaskOutStartValidation, os.path.join(save_path, file, 'MultMaskOutStartValidation'))
        torch.save(MultMaskInStartValidation, os.path.join(save_path, file, 'MultMaskInStartValidation'))
        InputTraining, OutputTraining, MaskTraining = MakeData(arg_simulateur, arg['len_in_full'], arg['len_in_window'],
                                                               arg['len_out'], arg['len_out_temp'], n_data_training)
        AddMaskOutEndTraining, MultMaskOutEndTraining, MultMaskOutStartTraining, MultMaskInStartTraining = MaskTraining
        torch.save(InputTraining, os.path.join(save_path, file, 'InputTraining'))
        torch.save(OutputTraining, os.path.join(save_path, file, 'OutputTraining'))
        torch.save(AddMaskOutEndTraining, os.path.join(save_path, file, 'AddMaskOutEndTraining'))
        torch.save(MultMaskOutEndTraining, os.path.join(save_path, file, 'MultMaskOutEndTraining'))
        torch.save(MultMaskOutStartTraining, os.path.join(save_path, file, 'MultMaskOutStartTraining'))
        torch.save(MultMaskInStartTraining, os.path.join(save_path, file, 'MultMaskInStartTraining'))



        return [[InputValidation, OutputValidation, MaskValidation],
                [InputTraining, OutputTraining, MaskTraining]]

if __name__ == '__main__':
    arg_simulateur = {
        'Fe_List': [5.1, 5, 4.9, 4.8],
        'Duree_max_impulsion': 4,
        'Seuil_mono': 10,
        'Seuil_harmo': 8,
        'Seuil_IM': 8,
        'Seuil_sensi_traitement': 6,
        'Seuil_sensi': 1,
        'Contraste_geneur': 0.2,
        'Nint': 500,
        'Contraste_geneur_2': 1,
        'M1_aveugle': 2,
        'M2_aveugle': 2,
        'M_local': 5,
        'N_DetEl': 12,
        'Seuil_ecart_freq': 5e-3,
        'Duree_maintien_max': 0.2,
        'N_mesureurs_max': 8,
        'PDW_tries': True,
    }

    MakeData(arg_simulateur, 50, 10, 15, 5, 5000, seed=0)