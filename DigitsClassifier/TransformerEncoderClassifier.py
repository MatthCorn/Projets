from Transformer.EncoderTransformer import EncoderLayer
import matplotlib.pyplot as plt
import idx2numpy
import torch
import torch.nn as nn
import os

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projet')

def MakeLabelSet(x):
    out = torch.zeros(x.shape[0],10)
    for i in range(len(x)):
        out[i,int(x[i])] = 1
    return out

TrainingImageSet = torch.tensor(idx2numpy.convert_from_file(local + r'\DigitsClassifier\Data\train-images.idx3-ubyte')).to(torch.float32)
TrainingLabels = torch.tensor(idx2numpy.convert_from_file(local + r'\DigitsClassifier\Data\train-labels.idx1-ubyte')).to(torch.int)
TrainingLabelSet = MakeLabelSet(TrainingLabels)
ValidationImageSet = torch.tensor(idx2numpy.convert_from_file(local + r'\DigitsClassifier\Data\t10k-images.idx3-ubyte')).to(torch.float32)
ValidationLabels = torch.tensor(idx2numpy.convert_from_file(local + r'\DigitsClassifier\Data\t10k-labels.idx1-ubyte')).to(torch.int)
ValidationLabelSet = MakeLabelSet(ValidationLabels)

d_model = 28
num_heads = 4
# d_head = d_model/num_heads = 7
# seq_len = 28
max_len = 64
d_out = 10

class ClassifierTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.FirstEncoder = EncoderLayer(d_model=d_model, d_att=d_model, num_heads=num_heads, WidthsFeedForward=[100, 100], max_len=64, MHADropout=0.1, FFDropout=0.05, masked=False)
        self.SecondEncoder = EncoderLayer(d_model=d_model, d_att=d_model, num_heads=num_heads, WidthsFeedForward=[100, 100], max_len=64, MHADropout=0.1, FFDropout=0.05, masked=False)
        self.FinalClassifier = nn.Linear(784, 10)

    def forward(self, x):
        # x.shape = (batch_size, seq_len, d_model)
        y = self.FirstEncoder(x)
        # y.shape = (batch_size, seq_len, d_model)
        y = self.SecondEncoder(y)
        # y.shape = (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = y.shape
        y = y.reshape(batch_size, -1)
        # y.shape = (batch_size, seq_len*d_model)
        y = self.FinalClassifier(y)
        # y.shape = (batch_size, 10)
        return y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = ClassifierTransformer().to(device)
TrainingImageSet = TrainingImageSet.to(device)
TrainingLabelSet = TrainingLabelSet.to(device)
ValidationImageSet = ValidationImageSet.to(device)
ValidationLabelSet = ValidationLabelSet.to(device)
ValidationLabels = ValidationLabels.to(device)

Batchs = [list(range(100*k,100*(k+1))) for k in range(600)]

optimizer = torch.optim.Adam(N.parameters(), weight_decay=0.05)

ErrorTrainingSet = []
AccuracyValidationSet = []

for i in range(100):
    print(i)
    for Batch in Batchs:
        optimizer.zero_grad(set_to_none=True)
        err = torch.norm(N(TrainingImageSet[Batch])-TrainingLabelSet[Batch])
        err.backward()
        optimizer.step()
    ErrorTrainingSet.append(float(err))
    AccuracyValidationSet.append(float(1 - torch.count_nonzero(torch.argmax(N(ValidationImageSet),dim=1)-ValidationLabels)/len(ValidationLabels)))

print(sum(p.numel() for p in N.parameters() if p.requires_grad))
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(AccuracyValidationSet); ax1.set_title("Précision sur l'ensemble de validation")
ax2.plot(ErrorTrainingSet); ax2.set_title("Erreur sur l'ensemble de test")
ax3.plot(torch.norm(N.FirstEncoder.MultiHeadAttention.Er,dim=1).cpu().detach().numpy()); ax3.set_title("Norme de Er sur le premier encodeur")
ax4.plot(torch.norm(N.SecondEncoder.MultiHeadAttention.Er,dim=1).cpu().detach().numpy()); ax4.set_title("Norme de Er sur le second encodeur")
