from Transformer.EasyFeedForward import FeedForward
import matplotlib.pyplot as plt
import idx2numpy
import torch
from tqdm import tqdm
import os

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')

def MakeLabelSet(x):
    out = torch.zeros(x.shape[0],10)
    for i in range(len(x)):
        out[i,int(x[i])] = 1
    return out

TrainingImageSet = torch.tensor(idx2numpy.convert_from_file(local + r'\DigitsClassifier\Data\train-images.idx3-ubyte')).reshape(60000,-1).to(torch.float32)
TrainingLabels = torch.tensor(idx2numpy.convert_from_file(local + r'\DigitsClassifier\Data\train-labels.idx1-ubyte')).to(torch.int)
TrainingLabelSet = MakeLabelSet(TrainingLabels)
ValidationImageSet = torch.tensor(idx2numpy.convert_from_file(local + r'\DigitsClassifier\Data\t10k-images.idx3-ubyte')).reshape(10000,-1).to(torch.float32)
ValidationLabels = torch.tensor(idx2numpy.convert_from_file(local + r'\DigitsClassifier\Data\t10k-labels.idx1-ubyte')).to(torch.int)
ValidationLabelSet = MakeLabelSet(ValidationLabels)

d_in = 784
d_out = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = FeedForward(d_in, d_out, widths = [200,50]).to(device)
TrainingImageSet = TrainingImageSet.to(device)
TrainingLabelSet = TrainingLabelSet.to(device)
ValidationImageSet = ValidationImageSet.to(device)
ValidationLabelSet = ValidationLabelSet.to(device)
ValidationLabels = ValidationLabels.to(device)

Batchs = [list(range(100*k, 100*(k+1))) for k in range(600)]

optimizer = torch.optim.Adam(N.parameters(), weight_decay=0.1, lr=1e-5)

ErrorTrainingSet = []
AccuracyValidationSet = []

for i in tqdm(range(50)):
    error = 0
    for Batch in Batchs:
        optimizer.zero_grad(set_to_none=True)
        err = torch.norm(N(TrainingImageSet[Batch])-TrainingLabelSet[Batch])
        err.backward()
        optimizer.step()
        error += float(err)
    ErrorTrainingSet.append(error)
    with torch.no_grad():
        AccuracyValidationSet.append(float(1 - torch.count_nonzero(torch.argmax(N(ValidationImageSet),dim=1)-ValidationLabels)/len(ValidationLabels)))

print(sum(p.numel() for p in N.parameters() if p.requires_grad))


fig, ((ax1, ax2)) = plt.subplots(2, 1)
ax1.plot(AccuracyValidationSet); ax1.set_title("Précision sur l'ensemble de validation")
ax2.plot(ErrorTrainingSet); ax2.set_title("Erreur sur l'ensemble de test")
plt.show()