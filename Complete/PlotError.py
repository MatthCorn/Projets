from Tools.XMLTools import loadXmlAsObj
import matplotlib.pyplot as plt
import numpy as np
import os

def Plot(path, std=False):
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


    ax11.plot(TrainingErrList, 'r', label="Ensemble d'entrainement")
    ax11.set_title('Erreur gobale')
    ax11.plot(ValidationErrList, 'b', label="Ensemble de Validation")
    ax11.legend(loc='upper right')
    ax11.set_ylim(bottom=0)

    ax12.plot([el[0]/Std[0] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement")
    ax12.set_title('Erreur sur TOA')
    ax12.plot([el[0]/Std[0] for el in ValidationErrTransList], 'b', label="Ensemble de Validation")
    ax12.legend(loc='upper right')
    ax12.set_ylim(bottom=0)

    ax21.plot([el[1]/Std[1] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement")
    ax21.set_title('Erreur sur LI')
    ax21.plot([el[1]/Std[1] for el in ValidationErrTransList], 'b', label="Ensemble de Validation")
    ax21.legend(loc='upper right')
    ax21.set_ylim(bottom=0)

    ax22.plot([el[2]/Std[2] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement")
    ax22.set_title('Erreur sur Niveau')
    ax22.plot([el[2]/Std[2] for el in ValidationErrTransList], 'b', label="Ensemble de Validation")
    ax22.legend(loc='upper right')
    ax22.set_ylim(bottom=0)

    ax31.plot([el[3]/Std[3] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement")
    ax31.set_title('Erreur sur FreqMin')
    ax31.plot([el[3]/Std[3] for el in ValidationErrTransList], 'b', label="Ensemble de Validation")
    ax31.legend(loc='upper right')
    ax31.set_ylim(bottom=0)

    ax32.plot([el[4]/Std[4] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement")
    ax32.set_title('Erreur sur FreqMax')
    ax32.plot([el[4]/Std[4] for el in ValidationErrTransList], 'b', label="Ensemble de Validation")
    ax32.legend(loc='upper right')
    ax32.set_ylim(bottom=0)

    plt.show()


if __name__ == '__main__':
    # local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'OneDrive', 'Documents', 'Python', 'Projets')
    local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'Documents', 'Python', 'Projets')

    folder = os.path.join('Complete', 'TypeTrackerInspired', 'Save', '2023-10-28__12-49', 'D_3', 'error')
    Plot(os.path.join(local, folder), std=False)

