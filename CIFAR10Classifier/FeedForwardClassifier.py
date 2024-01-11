from Projets.Transformer.EasyFeedForward import FeedForward
import matplotlib.pyplot as plt
import torch
import pickle

local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'OneDrive', 'Documents', 'Python', 'Projets')
# local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'Documents', 'Python', 'Projets')

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def MakeLabelSet(x):
    out = torch.zeros(x.shape[0],10)
    for i in range(len(x)):
        out[i,int(x[i])] = 1
    return out

def LoadBatch(i,device):
    dict = unpickle(local + r'\CIFAR10Classifier\Data\data_batch_' + str(i))
    data = torch.tensor(dict[b'data'])
    labels = MakeLabelSet(torch.tensor(dict[b'labels']))
    return data.to(device,torch.float32), labels.to(device)

def LoadValidation(device):
    dict = unpickle(local + r'\CIFAR10Classifier\Data\test_batch')
    data = torch.tensor(dict[b'data'])
    return data.to(device,torch.float32), torch.tensor(dict[b'labels']).to(device)

d_in = 3072
d_out = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = FeedForward(d_in, d_out, widths = [512,128,32]).to(device)

Batchs = [list(range(1000*k,1000*(k+1))) for k in range(10)]
MiniBatchs = [list(range(100*k,100*(k+1))) for k in range(10)]

optimizer = torch.optim.Adam(N.parameters(),weight_decay = 0.005)

ErrorTrainingSet = []
AccuracyValidationSet = []
ValidationImageSet, ValidationLabels = LoadValidation(device=torch.device('cpu'))
for i in range(200):
    print(i)
    CurrentError = 0
    for j in range(1,6):
        data, labels = LoadBatch(j, device=torch.device('cpu'))
        for Batch in Batchs:
            BatchData, BatchLabels = data[Batch].to(device), labels[Batch].to(device)
            for MiniBatch in MiniBatchs:
                optimizer.zero_grad()
                err = torch.norm(N(BatchData[MiniBatch]) - BatchLabels[MiniBatch])
                err.backward()
                optimizer.step()
                CurrentError += float(err)
    ErrorTrainingSet.append(CurrentError)
    if i%10 == 0:
        N.cpu()
        AccuracyValidationSet.append(float(1 - torch.count_nonzero(torch.argmax(N(ValidationImageSet),dim=1)-ValidationLabels)/len(ValidationLabels)))
        N.to(device)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(AccuracyValidationSet); ax1.set_title("Précision sur l'ensemble de validation")
ax2.plot(ErrorTrainingSet); ax2.set_title("Erreur sur l'ensemble de test")
plt.show()
print(sum(p.numel() for p in N.parameters() if p.requires_grad))