from Tools.XMLTools import loadXmlAsObj
import matplotlib.pyplot as plt
def Plot(path, eval=False):
    data = loadXmlAsObj(path)

    TrainingErrList, ValidationErrList = data['Training']['ErrList'], data['Validation']['ErrList']
    TrainingErrTransList, ValidationErrTransList = data['Training']['ErrTransList'], data['Validation']['ErrTransList']
    TrainingErrActList, ValidationErrActList = data['Training']['ErrActList'], data['Validation']['ErrActList']

    fig, ((ax11, ax12), (ax21, ax22), (ax31, ax32), (ax41, ax42)) = plt.subplots(4, 2)

    ax11.plot(TrainingErrList, 'r', label="Ensemble d'entrainement");
    ax11.set_title('Erreur gobale')
    ax11.plot(ValidationErrList, 'b', label="Ensemble de Validation");
    ax11.legend(loc='upper right')

    ax12.plot([el[0] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement");
    ax12.set_title('Erreur sur TOA')
    ax12.plot([el[0] for el in ValidationErrTransList], 'b', label="Ensemble de Validation");
    ax12.legend(loc='upper right')

    ax21.plot([el[1] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement");
    ax21.set_title('Erreur sur LI')
    ax21.plot([el[1] for el in ValidationErrTransList], 'b', label="Ensemble de Validation");
    ax21.legend(loc='upper right')

    ax22.plot([el[2] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement");
    ax22.set_title('Erreur sur Niveau')
    ax22.plot([el[2] for el in ValidationErrTransList], 'b', label="Ensemble de Validation");
    ax22.legend(loc='upper right')

    ax31.plot([el[3] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement");
    ax31.set_title('Erreur sur FreqMin')
    ax31.plot([el[3] for el in ValidationErrTransList], 'b', label="Ensemble de Validation");
    ax31.legend(loc='upper right')

    ax32.plot([el[4] for el in TrainingErrTransList], 'r', label="Ensemble d'entrainement");
    ax32.set_title('Erreur sur FreqMax')
    ax32.plot([el[4] for el in ValidationErrTransList], 'b', label="Ensemble de Validation");
    ax32.legend(loc='upper right')

    ax41.plot(TrainingErrActList, 'r', label="Ensemble d'entrainement");
    ax41.set_title("Erreur sur l'action")
    ax41.plot(ValidationErrActList, 'b', label="Ensemble de Validation");
    ax41.legend(loc='upper right')

    if eval:
        RealEvaluationList, CutEvaluationList = data['Evaluation']['Real'], data['Evaluation']['Cut']

        ax42.plot(RealEvaluationList, 'r', label="Erreur sur traduction réelle");
        ax42.set_title("Erreur sur traduction")
        ax42.plot(CutEvaluationList, 'b', label='Erreur sur traduction tronquée');
        ax42.legend(loc='upper right')

    plt.show()

if __name__ == '__main__':
    Plot(r'C:\\Users\\matth\\OneDrive\\Documents\\Python\\Projets\\FakeDigitalTwinTranslator\\Classic\\Save\\29-08-2023__20-57\\error', eval=True)