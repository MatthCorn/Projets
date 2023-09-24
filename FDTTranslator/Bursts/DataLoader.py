import os
import numpy as np
import torch

# Cette fonction ne prend pas en compte le chargement du mode évaluation pour l'instant
def FDTDataLoader(ListTypeData=[], local='', variables_dict={}, TypeBursts='FPII'):

    NewArg = []
    NewValue = []
    for TypeData in ['Validation', 'Training', 'Evaluation']:
        if TypeData in ListTypeData:
            Source = np.load(os.path.join(local, 'FDTTranslator', 'Bursts', TypeBursts, 'BurstsData', 'Data', TypeData, 'PulsesAnt_0.npy'))
            Translation = np.load(os.path.join(local, 'FDTTranslator', 'Bursts', TypeBursts, 'BurstsData', 'Data', TypeData, 'PDWsDCI_0.npy'))

            NewArg.append(TypeData + 'Source')
            NewValue.append(Source)
            NewArg.append(TypeData + 'Translation')
            NewValue.append(Translation)

    for i in range(len(NewArg)):
        variables_dict.__setitem__(NewArg[i], torch.tensor(NewValue[i]))

def LoadParam(dict, variables_dict):
    for key in dict.keys():
        variables_dict.__setitem__(key, dict[key])


