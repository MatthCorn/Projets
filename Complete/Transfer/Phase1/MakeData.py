'''
Pour cette phase, on pré-entraîne en mode auto-encoder :
Les entrées et les sorties sont sensiblement les mêmes mais en face de chaque PDW en sortie, on prédit les n_trackers PDWs suivants
'''

from Tools.XMLTools import loadXmlAsObj
import os
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import shutil
from tqdm import tqdm

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')

n_max_pulses = 10
n_PDWs_memory = 10
delta_t = 0.5
n_max_PDWs = 20
t_max = 35


# Cette fonction a pour rôle de transformer les phrases de source et translation en plein de petites
# phrases correspondant au découpage par salve temporelle de longueur delta_t
def Spliter(source, delta_t):
    new_source = []
    new_translation = []
    batch_len = len(source)
    for i in tqdm(range(batch_len)):
        source_sentence = source[i]
        translation_sentence = source[i]

        # On commence par découper la phrase de source et la phrase de translation sur les intervalles de taille delta_t
        splitted_source_sentence = []
        splitted_translation_sentence = []

        # Ces indices permettent de suivre les impulsions déjà ajoutées (comme elles sont triées par TOA)
        id_source = 0
        id_translation = 0
        t = 0
        while not (id_source == len(source_sentence) and id_translation == len(translation_sentence)):
            t += delta_t
            bursts_source = []
            while not id_source == len(source_sentence):
                pulse = source_sentence[id_source]
                if pulse[0] > t:
                    break
                pulse[0] -= t
                bursts_source.append(pulse)
                id_source += 1
            splitted_source_sentence.append(bursts_source)

            bursts_translation = []
            while not id_translation == len(translation_sentence):
                PDW = translation_sentence[id_translation]
                if PDW[0] > t:
                    break
                PDW[0] -= t
                bursts_translation.append(PDW)
                id_translation += 1
            splitted_translation_sentence.append(bursts_translation)

        while t < t_max:
            t += delta_t
            splitted_source_sentence.append([])
            splitted_translation_sentence.append([])

        # On reconstruit les phrases telles qu'elles seront données au traducteur
        remain = [[0] * 5] * n_max_pulses
        t = 0
        for bursts_source in splitted_source_sentence:
            t += delta_t
            remain = (remain + bursts_source)[-n_max_pulses:]
            new_source.append((np.array(remain) - [t, 0, 0, 0, 0]).tolist())

        remain = [[0] * 8] * n_PDWs_memory
        t = 0
        for bursts_translation in splitted_translation_sentence:
            remain = remain[-n_PDWs_memory:] + bursts_translation
            new_translation.append((np.array(remain) - [t, 0, 0, 0, 0, 0, 0, 0]).tolist())

    return new_source, new_translation

def FDTDataMaker():
    for type_data in ['Validation', 'Training']:
        pulses = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', type_data + 'PulsesAnt.xml'))
        PDWs = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', type_data + 'PDWsDCI.xml'))

        source = [
            [[pulse['TOA'], pulse['LI'], pulse['Level'], pulse['FreqStart'], pulse['FreqEnd']] for pulse in pulses_ant]
            for pulses_ant in pulses]
        translation = [[[PDW['TOA'], PDW['LI'], PDW['Level'], PDW['FreqMin'], PDW['FreqMax'], int('CW' in PDW['flags']),
                         int('TroncAv' in PDW['flags']),
                         int(len(PDW['flags']) == 0)] for PDW in PDWs_DCI] for PDWs_DCI in PDWs]

        # Ici les données sont founies en batch
        # source est une liste de séquence d'impulsions de même longueur
        # translation est une liste de séquence de PDWs, elles peuvent être de longueures différentes


        source, translation = Spliter(source, translation, delta_t)
        # On transforme ces listes en tenseur
        source = torch.tensor(source)
        # source.shape = (batch_size, len_input, d_input)

        # Rajoute des 0 à la fin des scénarios de PDWs pour qu'ils aient toutes la même longueure
        translation = pad_sequence([torch.tensor(el) for el in translation], batch_first=True)

        _, temp_len, _ = translation.shape
        translation = F.pad(translation, (0, 0, 0, n_max_PDWs - temp_len))
        # translation.shape = (batch_size, n_max_PDWs, d_target + num_flags)

        WriteBatchs(source=source, translation=translation, type_data=type_data)


def WriteBatchs(source, translation, type_data):
    save_path = os.path.join(local, 'Complete', 'Data')
    try:
        shutil.rmtree(os.path.join(save_path, type_data))
    except:
        None

    os.mkdir(os.path.join(save_path, type_data))

    name_file_source = os.path.join(save_path, type_data, 'PulsesAnt.npy')
    name_file_translation = os.path.join(save_path, type_data, 'PDWsDCI.npy')
    np.save(name_file_source, source.numpy())
    np.save(name_file_translation, translation.numpy())

if __name__ == '__main__':
    FDTDataMaker()
