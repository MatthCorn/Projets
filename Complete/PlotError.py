from Tools.XMLTools import loadXmlAsObj
import matplotlib.pyplot as plt
import numpy as np
import torch
import os



def Plot(path, std=False, smoothing_factor=1):
    data = loadXmlAsObj(path)

    TrainingErrList, ValidationErrList = data['Training']['ErrList'], data['Validation']['ErrList']
    TrainingErrTransList, ValidationErrTransList = data['Training']['ErrTransList'], data['Validation']['ErrTransList']

    fig, ((ax11, ax12), (ax21, ax22), (ax31, ax32)) = plt.subplots(3, 2)

    Std = [1.]*len(TrainingErrTransList[0])
    if std:
        # On calcule l'écart type et on trace sur chaque subplot
        TransPath = os.path.join(path.split('Type')[0], 'Data', 'D_0.5', 'Training', 'PDWsDCI.npy')
        Translation = np.load(TransPath)
        Std = np.std(Translation, axis=(0, 1))


    ax11.plot(smooth(TrainingErrList, smoothing_factor), 'r', label="Ensemble d'entrainement")
    ax11.set_title('Erreur gobale')
    ax11.plot(smooth(ValidationErrList, smoothing_factor), 'b', label="Ensemble de Validation")
    ax11.legend(loc='upper right')
    ax11.set_ylim(bottom=0)

    ax12.plot(smooth([el[0]/Std[0] for el in TrainingErrTransList], smoothing_factor), 'r', label="Ensemble d'entrainement")
    ax12.set_title('Erreur sur TOA')
    ax12.plot(smooth([el[0]/Std[0] for el in ValidationErrTransList], smoothing_factor), 'b', label="Ensemble de Validation")
    ax12.legend(loc='upper right')
    ax12.set_ylim(bottom=0)

    ax21.plot(smooth([el[1]/Std[1] for el in TrainingErrTransList], smoothing_factor), 'r', label="Ensemble d'entrainement")
    ax21.set_title('Erreur sur LI')
    ax21.plot(smooth([el[1]/Std[1] for el in ValidationErrTransList], smoothing_factor), 'b', label="Ensemble de Validation")
    ax21.legend(loc='upper right')
    ax21.set_ylim(bottom=0)

    ax22.plot(smooth([el[2]/Std[2] for el in TrainingErrTransList], smoothing_factor), 'r', label="Ensemble d'entrainement")
    ax22.set_title('Erreur sur Niveau')
    ax22.plot(smooth([el[2]/Std[2] for el in ValidationErrTransList], smoothing_factor), 'b', label="Ensemble de Validation")
    ax22.legend(loc='upper right')
    ax22.set_ylim(bottom=0)

    ax31.plot(smooth([el[3]/Std[3] for el in TrainingErrTransList], smoothing_factor), 'r', label="Ensemble d'entrainement")
    ax31.set_title('Erreur sur FreqMoy')
    ax31.plot(smooth([el[3]/Std[3] for el in ValidationErrTransList], smoothing_factor), 'b', label="Ensemble de Validation")
    ax31.legend(loc='upper right')
    ax31.set_ylim(bottom=0)

    ax32.plot(smooth([el[4]/Std[4] for el in TrainingErrTransList], smoothing_factor), 'r', label="Ensemble d'entrainement")
    ax32.set_title('Erreur sur DeltaFreq')
    ax32.plot(smooth([el[4]/Std[4] for el in ValidationErrTransList], smoothing_factor), 'b', label="Ensemble de Validation")
    ax32.legend(loc='upper right')
    ax32.set_ylim(bottom=0)

    plt.show()

def Plot_2(path, std=False, smoothing_factor=1):
    data = loadXmlAsObj(path)

    TrainingErrList, ValidationErrList = data['Training'], data['Validation']

    fig, ax = plt.subplots(1, 1)

    Std = [1.]*len(TrainingErrList)
    if std:
        # On calcule l'écart type et on trace sur chaque subplot
        TransPath = os.path.join(path.split('Save')[0], 'Data', 'config0', 'OutputValidation')
        Translation = torch.load(TransPath)
        Std = torch.std(Translation) * np.array(Std)

    ax.plot(smooth(TrainingErrList, smoothing_factor), 'r', label="Training")
    ax.plot(smooth(ValidationErrList, smoothing_factor), 'b', label="Validation")
    ax.plot(Std, 'black')
    ax.set_title('Error through epochs')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)

    plt.show()

def smooth(Li, k):
    resu = []
    for i in range(k):
        resu.append(sum(Li[:i+1])/(i+1))
    for i in range(k, len(Li)):
        resu.append(sum(Li[i-k+1:i+1])/k)
    return resu

if __name__ == '__main__':
    local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')

    folder = os.path.join('Base', 'Save', '2024-10-10__13-29', 'error')
    Plot_2(os.path.join(local, folder), std=True, smoothing_factor=1)

