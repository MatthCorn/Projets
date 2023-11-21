from Tools.XMLTools import loadXmlAsObj
import os
import torch
import numpy as np
from tqdm import tqdm
from FakeDigitalTwin.SciptData import MakeSets
from FakeDigitalTwin.Experience import MakeData
from Complete.PreEmbedding import FeaturesAndScaling

# local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'OneDrive', 'Documents', 'Python', 'Projets')
local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'Documents', 'Python', 'Projets')

# Temps de maintien max d'un mesureur sans voir son impulsion
holding_time = 0.5
threshold = -7
n_max_pulses = 32
n_PDWs_memory = 10
delta_t = 3
t_max = 105

# Cette fonction donne un majorant de la date de publication d'un PDW donné
def TimeRelease(PDW):
    # PDW est une liste de la forme [TOA: float, LI: float, Level: float, FreqMin: float, FreqMax: float, flag1: int, flag2: int, flag3: int]
    maj = PDW[0] + PDW[1] + holding_time
    return maj

# Cette fonction a pour rôle de transformer les phrases de source et translation en plein de petites
# phrases correspondant au découpage par salve temporelle de longueur delta_t
def Spliter(source, translation, delta_t):
    new_source = []
    new_translation = []
    batch_len = len(source)
    for i in tqdm(range(batch_len)):
        source_sentence = source[i]
        translation_sentence = translation[i]
        # Il faut trier les PDWs par date de publication
        translation_sentence.sort(key=TimeRelease)

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
                bursts_source.append(pulse + [1, 0, 0])
                id_source += 1
            splitted_source_sentence.append(bursts_source)

            bursts_translation = []
            while not id_translation == len(translation_sentence):
                PDW = translation_sentence[id_translation]
                if PDW[0] > t:
                    break
                PDW[0] -= t
                bursts_translation.append(PDW + [1, 0, 0])
                id_translation += 1
            splitted_translation_sentence.append(bursts_translation)

        while t < t_max:
            t += delta_t
            splitted_source_sentence.append([])
            splitted_translation_sentence.append([])



        # On reconstruit les phrases telles qu'elles seront données au traducteur
        remain = [[0] * 5 + [0, 1, 0]] * (n_max_pulses - 1) + [[0] * 5 + [0, 0, 1]]
        t = 0
        for bursts_source in splitted_source_sentence:
            t += delta_t
            remain = (remain + bursts_source)[-n_max_pulses:]
            new_source.append((np.array(remain) - ([t, 0, 0, 0, 0] + [0, 0, 0])).tolist())

        remain = [[0] * 8 + [0, 1, 0]] * (n_PDWs_memory - 1) + [[0] * 8 + [0, 0, 1]]
        t = 0
        for bursts_translation in splitted_translation_sentence:
            remain = remain[-n_PDWs_memory:] + bursts_translation
            new_translation.append((np.array(remain) - ([t, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 0])).tolist())

    return new_source, new_translation

def FDTDataMaker(list_density):
    for density in list_density:
        print('density :', str(density))
        MakeSets(density)

        type_data = 'Training'

        pulses = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', type_data + 'PulsesAnt.xml'))
        PDWs = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', type_data + 'PDWsDCI.xml'))

        source = [
            [[pulse['TOA'], pulse['LI'], pulse['Level'], pulse['FreqStart'], pulse['FreqEnd']] for pulse in pulses_ant]
            for pulses_ant in pulses]
        translation = [[[PDW['TOA'], PDW['LI'], PDW['Level'], (PDW['FreqMin']+PDW['FreqMax'])/2, (PDW['FreqMax']-PDW['FreqMin'])/2, int('CW' in PDW['flags']),
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
        translation = pad_sequence(translation)

        Write(source=source, translation=translation, type_data=type_data, density=density)

def pad_sequence(l):
    max_len = max(len(sublist) for sublist in l)  # Trouver la longueur maximale des sous-listes
    l = [sublist + [[0.] * 8 + [0., 1., 0.]] * (max_len - len(sublist)) for sublist in l]
    return torch.tensor(l)

def Write(source, translation, type_data, density):
    save_path = os.path.join(local, 'Complete', 'Data')
    try:
        os.mkdir(os.path.join(save_path, 'D_'+str(density)))
    except:
        None

    os.mkdir(os.path.join(save_path, 'D_'+str(density), type_data))

    name_file_source = os.path.join(save_path, 'D_'+str(density), type_data, 'PulsesAnt.npy')
    name_file_translation = os.path.join(save_path, 'D_'+str(density), type_data, 'PDWsDCI.npy')
    np.save(name_file_source, source.numpy())
    np.save(name_file_translation, translation.numpy())


if False:
    for dir in os.listdir(os.path.join(local, 'Complete', 'Data')):
        s = torch.tensor(np.load(os.path.join(local, 'Complete', 'Data', dir, 'Training', 'PDWsDCI.npy')))
        print(dir + '/n')
        print(torch.norm(s, dim=(0, 2), p=1))

# Cette fonction ne prend pas en compte le chargement du mode évaluation pour l'instant
def FDTDataLoader(path='', len_target=30):

    type_data = 'Validation'
    validation_source = torch.tensor(np.load(os.path.join(path, type_data, 'PulsesAnt.npy')))
    validation_translation = torch.tensor(np.load(os.path.join(path, type_data, 'PDWsDCI.npy')))
    batch_size, temp_len, _ = validation_translation.shape
    pad = torch.tensor([[0.] * 8 + [0., 1., 0.]] * (len_target - temp_len)).unsqueeze(0).expand(batch_size, -1, -1)
    validation_translation = torch.cat((validation_translation, pad), dim=1)

    type_data = 'Training'
    training_source = torch.tensor(np.load(os.path.join(path, type_data, 'PulsesAnt.npy')))
    training_translation = torch.tensor(np.load(os.path.join(path, type_data, 'PDWsDCI.npy')))
    batch_size, temp_len, _ = training_translation.shape
    pad = torch.tensor([[0.] * 8 + [0., 1., 0.]] * (len_target - temp_len)).unsqueeze(0).expand(batch_size, -1, -1)
    training_translation = torch.cat((training_translation, pad), dim=1)

    return validation_source, validation_translation, training_source, training_translation

# Cette fonction permet de créer l'ensemble d'entrainement sans passer par l'écriture des données en format xml par le FDT, on gagne énormement de temps
def FastDataGen(list_density, batch_size={'Training': 6000, 'Validation': 300}):
    for density in list_density:

        for key in batch_size.keys():

            pulses, PDWs = MakeData(Batch_size=batch_size[key], seed=None, density=density, name=None, return_data=True)

            source = [
                [[pulse['TOA'], pulse['LI'], pulse['Level'], pulse['FreqStart'], pulse['FreqEnd']] for pulse in pulses_ant]
                for pulses_ant in pulses]
            translation = [[[PDW['TOA'], PDW['LI'], PDW['Level'], (PDW['FreqMin']+PDW['FreqMax'])/2, (PDW['FreqMax']-PDW['FreqMin'])/2, int('CW' in PDW['flags']),
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
            translation = pad_sequence(translation)

            Write(source=source, translation=translation, type_data=key, density=density)


def MakeWeights(batch_size, density, threshold=threshold):
    pulses, PDWs = MakeData(Batch_size=batch_size, seed=None, density=density, name=None, return_data=True)

    pulse_pre_embedding = FeaturesAndScaling(threshold, type='source')
    PDW_pre_embedding = FeaturesAndScaling(threshold, type='target')

    source = [
        [[pulse['TOA'], pulse['LI'], pulse['Level'], pulse['FreqStart'], pulse['FreqEnd']] for pulse in pulses_ant]
        for pulses_ant in pulses]
    translation = [[[PDW['TOA'], PDW['LI'], PDW['Level'], PDW['FreqMin'],
                     PDW['FreqMax'], int('CW' in PDW['flags']), int('TroncAv' in PDW['flags']),
                     int(len(PDW['flags']) == 0)] for PDW in PDWs_DCI] for PDWs_DCI in PDWs]

    # Ici les données sont founies en batch
    # source est une liste de séquence d'impulsions de même longueur
    # translation est une liste de séquence de PDWs, elles peuvent être de longueures différentes

    source, translation = Spliter(source, translation, delta_t)

    temp = []
    for sentence in source:
        for word in sentence:
            if word[5] == 1:
                temp.append(word)
    source = torch.tensor(temp)

    temp = []
    for sentence in translation:
        for word in sentence:
            if word[8] == 1:
                temp.append(word)
    translation = torch.tensor(temp)

    source = pulse_pre_embedding(source).numpy()
    translation = PDW_pre_embedding(translation).numpy()

    source_weights = np.std(np.array(source), axis=0) + np.abs(np.mean(np.array(source), axis=0))

    translation_weights = np.std(np.array(translation), axis=0) + np.abs(np.mean(np.array(translation), axis=0))


    print(source_weights)
    print(translation_weights)

if __name__ == '__main__':
    # FDTDataMaker(list_density=[0.3, 0.4, 0.5, 0.7, 0.9, 1.2, 1.5, 1.8, 2.2, 2.6, 3])
    # FastDataGen(list_density=[0.3, 0.5, 0.9, 1.5, 2.2, 3], batch_size={'Training': 30000, 'Validation': 300})
    MakeWeights(batch_size=1000, density=3)
