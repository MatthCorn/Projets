from Tools.XMLTools import loadXmlAsObj
import matplotlib.pyplot as plt
import numpy as np
import os

def Plot(path, std=False):
    data = loadXmlAsObj(path)

    TrainingErrList, ValidationErrList = data['Training']['ErrList'], data['Validation']['ErrList']
    TrainingErrTransList, ValidationErrTransList = data['Training']['ErrTransList'], data['Validation']['ErrTransList']

    fig, ((ax11, ax12), (ax21, ax22), (ax31, ax32)) = plt.subplots(3, 2)

    ax11.plot(TrainingErrList, 'r', label="Ensemble d'entrainement")
    ax11.set_title('Erreur gobale')
    ax11.plot(ValidationErrList, 'b', label="Ensemble de Validation")
    ax11.legend(loc='upper right')
    ax11.set_ylim(bottom=0)

    ax12.plot([el[0] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement")
    ax12.set_title('Erreur sur TOA')
    ax12.plot([el[0] for el in ValidationErrTransList], 'b', label="Ensemble de Validation")
    ax12.legend(loc='upper right')
    ax12.set_ylim(bottom=0)

    ax21.plot([el[1] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement")
    ax21.set_title('Erreur sur LI')
    ax21.plot([el[1] for el in ValidationErrTransList], 'b', label="Ensemble de Validation")
    ax21.legend(loc='upper right')
    ax21.set_ylim(bottom=0)

    ax22.plot([el[2] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement")
    ax22.set_title('Erreur sur Niveau')
    ax22.plot([el[2] for el in ValidationErrTransList], 'b', label="Ensemble de Validation")
    ax22.legend(loc='upper right')
    ax22.set_ylim(bottom=0)

    ax31.plot([el[3] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement")
    ax31.set_title('Erreur sur FreqStart')
    ax31.plot([el[3] for el in ValidationErrTransList], 'b', label="Ensemble de Validation")
    ax31.legend(loc='upper right')
    ax31.set_ylim(bottom=0)

    ax32.plot([el[4] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement")
    ax32.set_title('Erreur sur FreqEnd')
    ax32.plot([el[4] for el in ValidationErrTransList], 'b', label="Ensemble de Validation")
    ax32.legend(loc='upper right')
    ax32.set_ylim(bottom=0)

    if std:
        # On calcule l'écart type et on trace sur chaque subplot
        TransPath = os.path.join(path.split('Save')[0], 'BurstsData', 'Data', 'Training', 'PDWsDCI_0.npy')
        Translation = np.load(TransPath)
        Std = np.std(Translation, axis=(0, 1))

        ax12.plot([Std[0]]*len(TrainingErrList), 'k')
        ax22.plot([Std[2]]*len(TrainingErrList), 'k')
        ax21.plot([Std[1]]*len(TrainingErrList), 'k')
        ax31.plot([Std[3]]*len(TrainingErrList), 'k')
        ax32.plot([Std[4]]*len(TrainingErrList), 'k')

    plt.show()

def PlotPropError(path, deg=4):
    PropError = loadXmlAsObj(path)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for key in PropError.keys():
        for i in range(2):
            Pos = PropError[key][i][0]
            Diff = PropError[key][i][1]

            p = np.poly1d(np.polyfit(Pos, Diff, deg=deg))
            x = np.linspace(0, 1, 100)
            y = p(x)

            if i == 0:
                ax1.plot(x, y, label=key)
            else:
                ax2.plot(x, y, label=key)

    ax1.set_title('Propagation of error on real translation')
    ax1.legend(loc='upper right')
    ax1.set_ylim(bottom=0)

    ax2.set_title('Propagation of error on troncated translation')
    ax2.legend(loc='upper right')
    ax2.set_ylim(bottom=0)

    plt.show()


if __name__ == '__main__':
    local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'OneDrive', 'Documents', 'Python', 'Projets')
    # local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'Documents', 'Python', 'Projets')

    folder = os.path.join('StepByStep', 'S1', 'Save', '2023-09-26__11-53', 'error')
    Plot(os.path.join(local, folder), std=True)

