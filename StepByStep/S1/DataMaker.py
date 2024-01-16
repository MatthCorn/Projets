import torch
import numpy as np
from tqdm import tqdm
import os
import shutil

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')

def MakeData(batch_size, decay=0.8):
    Source = torch.randn(size=(batch_size, 10, 5))
    Source -= torch.min(Source)
    for i in range(1, 10):
        Source[:, i, 0] += Source[:, i-1, 0]
    Translation = Source.clone()

    for i in tqdm(range(batch_size)):
        TgtSentence = Translation[i]
        # TgtSentence.shape = (10, 5)
        for j in range(1, 10):
            W0 = TgtSentence[j-1]
            W1 = TgtSentence[j]
            Level = W0[1]*(decay**(W1[0]-W0[0]))

            if Level > W1[1]:
                W1[2:] = (W1[2:] - W0[2:])/2
                W1[1] = Level
            else:
                W1[2:] = (W1[2:] + W0[2:]) / 2

            TgtSentence[j] = W1

        Translation[i] = TgtSentence

    return Source, Translation

def WriteBatchs(batch_size, TypeData):
    Source, Translation = MakeData(batch_size)
    save_path = os.path.join(local, 'StepByStep', 'S1', 'Data')
    try:
        shutil.rmtree(os.path.join(save_path, TypeData))
    except:
        None

    os.mkdir(os.path.join(save_path, TypeData))

    Source = Source.numpy()
    Translation = Translation.numpy()
    SourceFName = os.path.join(save_path, TypeData, 'PulsesAnt.npy')
    TranslationFName = os.path.join(save_path, TypeData, 'PDWsDCI.npy')
    np.save(SourceFName, Source)
    np.save(TranslationFName, Translation)

if __name__ == '__main__':
    WriteBatchs(1000000, 'Training')
    WriteBatchs(6000, 'Validation')

