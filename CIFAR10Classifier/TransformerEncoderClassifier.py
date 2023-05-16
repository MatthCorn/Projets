from Transformer.EncoderTransformer import EncoderLayer
from Transformer.EasyFeedForward import FeedForward
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle

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
    dict = unpickle(r'C:\Users\Matthieu\Documents\Python\Projets\CIFAR10Classifier\data_batch_' + str(i))
    data = torch.tensor(dict[b'data'])
    # data.shape = (BatchSize,3*dimx*dimy)
    BatchSize, temp = data.shape
    seq_len = int((temp/3) ** 0.5)
    data = data.reshape(BatchSize,3,seq_len,seq_len)
    data = data.permute(0,2,3,1)
    data = data.reshape(BatchSize,seq_len,-1)
    return data.to(device,torch.float32), MakeLabelSet(dict[b'labels']).to(device,torch.float32)

def LoadValidation(device):
    dict = unpickle(r'C:\Users\Matthieu\Documents\Python\Projets\CIFAR10Classifier\test_batch')
    data = torch.tensor(dict[b'data'])
    # data.shape = (BatchSize,3*dimx*dimy)
    BatchSize, temp = data.shape
    seq_len = int((temp/3) ** 0.5)
    data = data.reshape(BatchSize,3,seq_len,seq_len)
    data = data.permute(0,2,3,1)
    data = data.reshape(BatchSize,seq_len,-1)
    return data.to(device,torch.float32), torch.tensor(dict[b'labels']).to(device)

d_model = 96
num_heads = 12
# d_head = d_model/num_heads = 8
seq_len = 32
max_len = 64
d_out = 10

class ClassifierTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.FirstEncoder = EncoderLayer(d_model, num_heads, WidthsFeedForward=[100,100], max_len=80, MHADropout=0.1, FFDropout=0.05, masked=False)
        self.SecondEncoder = EncoderLayer(d_model, num_heads, WidthsFeedForward=[100, 100], max_len=80, MHADropout=0.1, FFDropout=0.05, masked=False)
        self.ThirdEncoder = EncoderLayer(d_model, num_heads, WidthsFeedForward=[100, 100], max_len=80, MHADropout=0.1, FFDropout=0.05, masked=False)
        self.DimDownScaler = FeedForward(d_model, 16, widths=[16], dropout=0.05)
        self.FinalClassifier = FeedForward(seq_len*16, 10, widths=[128,32], dropout=0.05)

    def forward(self,x):
        # x.shape = (batch_size, seq_len, d_model)
        y = self.FirstEncoder(x)
        # y.shape = (batch_size, seq_len, d_model)
        y = self.SecondEncoder(y)
        # y.shape = (batch_size, seq_len, d_model)
        y = self.ThirdEncoder(y)
        # y.shape = (batch_size, seq_len, d_model)
        y = self.DimDownScaler(y)
        # y.shape = (batch_size, seq_len, 16)
        batch_size, _, _ = y.shape
        y = y.reshape(batch_size, -1)
        # y.shape = (batch_size, seq_len*16)
        y = self.FinalClassifier(y)
        # y.shape = (batch_size, 10)
        return y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = ClassifierTransformer().to(device)

MiniBatchs = [list(range(100*k,100*(k+1))) for k in range(10)]

optimizer = torch.optim.Adam(N.parameters(),weight_decay = 0.05)

ErrorTrainingSet = []
AccuracyValidationSet = []
ValidationImageSet, ValidationLabels = LoadValidation(device=torch.device('cpu'))

LittleBatchs = [list(range(1000*k,1000*(k+1))) for k in range(10)]

for i in range(100):
    print('i = ' + str(i))
    for j in range(1,6):
        BatchData, BatchLabels = LoadBatch(j,device=torch.device('cpu'))
        for LittleBatch in LittleBatchs:
            data, labels = BatchData[LittleBatch].to(device), BatchLabels[LittleBatch].to(device)
            for MiniBatch in MiniBatchs:

                optimizer.zero_grad()
                err = torch.norm(N(data[MiniBatch]) - labels[MiniBatch])
                err.backward()
                optimizer.step()
            ErrorTrainingSet.append(float(err))
    # AccuracyValidationSet.append(float(1 - torch.count_nonzero(torch.argmax(N(ValidationImageSet.to(device)),dim=1)-ValidationLabels.to(device))/len(ValidationLabels.to(device))))


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# ax1.plot(AccuracyValidationSet); ax1.set_title("Précision sur l'ensemble de validation")
ax2.plot(ErrorTrainingSet); ax2.set_title("Erreur sur l'ensemble de test")
ax3.plot(torch.norm(N.FirstEncoder.MultiHeadAttention.Er,dim=1).cpu().detach().numpy()); ax3.set_title("Norme de Er sur le premier encodeur")
ax4.plot(torch.norm(N.SecondEncoder.MultiHeadAttention.Er,dim=1).cpu().detach().numpy()); ax4.set_title("Norme de Er sur le second encodeur")
plt.show()

print(sum(p.numel() for p in N.parameters() if p.requires_grad))