from Perceiver.EncoderPerceiver import EncoderIO
from Transformer.EasyFeedForward import FeedForward
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle

local = r'C:\Users\matth\OneDrive\Documents\Python\Projets'
# local = r'C:\Users\Matthieu\Documents\Python\Projets'
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def MakeLabelSet(x):
    out = torch.zeros(len(x),10)
    for i in range(len(x)):
        out[i,int(x[i])] = 1
    return out

def PlotImage(i,data):
    im = torch.tensor(data[i]).reshape(3,32,32)
    im = im.permute(1,2,0)
    plt.imshow(im.numpy())
    plt.show()

def LoadBatch(i,device):
    dict = unpickle(local + r'\CIFAR10Classifier\data_batch_' + str(i))
    data = torch.tensor(dict[b'data'])
    # data.shape = (BatchSize,3*dimx*dimy)
    BatchSize, temp = data.shape
    seq_len = int((temp/3) ** 0.5)
    data = data.reshape(BatchSize,3,seq_len,seq_len)
    data = data.reshape(BatchSize,-1,seq_len)
    return data.to(device,torch.float32), MakeLabelSet(dict[b'labels']).to(device,torch.float32)

def LoadValidation(device):
    dict = unpickle(local + r'\CIFAR10Classifier\test_batch')
    data = torch.tensor(dict[b'data'])
    # data.shape = (BatchSize,3*dimx*dimy)
    BatchSize, temp = data.shape
    seq_len = int((temp/3) ** 0.5)
    data = data.reshape(BatchSize,3,seq_len,seq_len)
    data = data.permute(0,2,3,1)
    data = data.reshape(BatchSize,-1,seq_len)
    return data.to(device,torch.float32), torch.tensor(dict[b'labels']).to(device)

d_latent = 32
d_att = 32
d_input = 32
num_heads = 4
# d_head = d_model/num_heads = 8
input_len = 96
latent_len = 16
max_len = 64
d_out = 10

class ClassifierTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = EncoderIO(d_latent=d_latent, d_input=d_input, d_att=d_att, num_heads=num_heads, latent_len=latent_len)
        self.FinalClassifier = FeedForward(d_in=latent_len*d_latent, d_out=10, widths=[256, 64, 32], dropout=0.05)

    def forward(self, x):
        # x_input.shape = (batch_size, input_len, d_input)
        y = self.Encoder(x)
        # y.shape = (batch_size, latent_len, d_latent)
        batch_size, _, _ = y.shape
        y = y.reshape(batch_size, -1)
        # y.shape = (batch_size, latent_len*d_latent)
        y = self.FinalClassifier(y)
        # y.shape = (batch_size, 10)
        return y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = ClassifierTransformer().to(device)

MiniBatchs = [list(range(100*k,100*(k+1))) for k in range(10)]

optimizer = torch.optim.Adam(N.parameters(),weight_decay = 0.0001)

ErrorTrainingSet = []
AccuracyValidationSet = []
ValidationImageSet, ValidationLabels = LoadValidation(device=torch.device('cpu'))

LittleBatchs = [list(range(1000*k,1000*(k+1))) for k in range(10)]

for i in range(100):
    print('i = ' + str(i))
    CurrentError = 0
    for j in range(1,6):
        BatchData, BatchLabels = LoadBatch(j,device=torch.device('cpu'))
        for LittleBatch in LittleBatchs:
            data, labels = BatchData[LittleBatch].to(device), BatchLabels[LittleBatch].to(device)
            for MiniBatch in MiniBatchs:

                optimizer.zero_grad()
                err = torch.norm(N(data[MiniBatch]) - labels[MiniBatch])
                err.backward()
                optimizer.step()
                CurrentError += float(err)
    ErrorTrainingSet.append(CurrentError)
    if i % 5 == 0:
        Err = 0
        for LittleBatch in LittleBatchs:
            data, labels = ValidationImageSet[LittleBatch].to(device), ValidationLabels[LittleBatch].to(device)
            Err += torch.count_nonzero(torch.argmax(N(data), dim=1) - labels)
        AccuracyValidationSet.append(float(1 - Err / len(ValidationLabels)))

fig, ((ax1, ax2)) = plt.subplots(2, 1)
ax1.plot(AccuracyValidationSet); ax1.set_title("Précision sur l'ensemble de validation")
ax2.plot(ErrorTrainingSet); ax2.set_title("Erreur sur l'ensemble de test")
plt.show()

print(sum(p.numel() for p in N.parameters() if p.requires_grad))