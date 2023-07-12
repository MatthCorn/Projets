from Perceiver.EncoderPerceiver import EncoderIO
from Transformer.EasyFeedForward import FeedForward
from CIFAR10Classifier.Config import config, MakeLabelSet
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

local = r'C:\Users\matth\OneDrive\Documents\Python\Projets'
# local = r'C:\Users\Matthieu\Documents\Python\Projets'

LocalConfig = config(config=3)
LocalConfig.AddParam(d_latent=8, d_att=8, num_heads=1, latent_len=32, max_len=64, d_out=10)


class ClassifierPerceiver(nn.Module):
    def __init__(self, d_latent=LocalConfig.d_latent, d_input=LocalConfig.d_input, d_att=LocalConfig.d_att, relative=True,
                 latent_len=LocalConfig.latent_len, d_out=LocalConfig.d_out, num_heads=LocalConfig.num_heads):
        super().__init__()
        self.Encoder = EncoderIO(d_latent=d_latent, d_input=d_input, d_att=d_att, num_heads=num_heads, latent_len=latent_len, SelfAttentionDepth=2, relative=relative)
        self.FinalClassifier = FeedForward(d_in=d_latent, d_out=d_out, widths=[16], dropout=0.05)

    def forward(self, x):
        # x_input.shape = (batch_size, input_len, d_input)
        y = self.Encoder(x)
        # y.shape = (batch_size, latent_len, d_latent)
        y = torch.mean(y, dim=1)
        # y.shape = (batch_size, d_latent)
        y = self.FinalClassifier(y)
        # y.shape = (batch_size, 10)
        return y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
type = torch.float16

N = ClassifierPerceiver(relative=True).to(device)

MiniBatchs = [list(range(100*k, 100*(k+1))) for k in range(5)]

optimizer = torch.optim.Adam(N.parameters(), weight_decay=1e-6, lr=1e-3)
scaler = GradScaler()
loss = nn.CrossEntropyLoss()

ErrorTrainingSet = []
AccuracyTrainingSet = []
ErrorValidationSet = []
AccuracyValidationSet = []
ValidationEpoch = []
ValidationImageSet, ValidationLabels = LocalConfig.LoadValidation(local)

LittleBatchs = [list(range(500*k, 500*(k+1))) for k in range(20)]

for i in tqdm(range(100)):
    CurrentError = 0
    AccErr = 0
    for j in range(1, 6):
        BatchData, BatchLabels = LocalConfig.LoadBatch(j, local)
        for LittleBatch in LittleBatchs:
            data, labels = BatchData[LittleBatch].to(device), BatchLabels[LittleBatch].to(device)
            for MiniBatch in MiniBatchs:
                with torch.autocast(device_type='cuda', dtype=type):
                    out = N(data[MiniBatch])
                    err = loss(out, labels[MiniBatch])
                    # err = torch.norm(N(data[MiniBatch]) - MakeLabelSet(labels[MiniBatch]))
                    AccErr += torch.count_nonzero(torch.argmax(out, dim=1) - labels[MiniBatch])
                scaler.scale(err).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                CurrentError += float(err)
    ErrorTrainingSet.append(CurrentError)
    AccuracyTrainingSet.append(float(1 - AccErr / (6*len(BatchLabels))))
    if i % 1 == 0:
        ValidationEpoch.append(i)
        AccErr = 0
        Err = 0
        for LittleBatch in LittleBatchs:
            data, labels = ValidationImageSet[LittleBatch].to(device), ValidationLabels[LittleBatch].to(device)
            for MiniBatch in MiniBatchs:
                with torch.autocast(device_type='cuda', dtype=type):
                    out = N(data[MiniBatch])
                    Err += 6*float(loss(out, labels[MiniBatch]))
                    # Err += 6*float(torch.norm(N(data[MiniBatch]) - MakeLabelSet(labels[MiniBatch])))
                    AccErr += torch.count_nonzero(torch.argmax(out, dim=1) - labels[MiniBatch])
        AccuracyValidationSet.append(float(1 - AccErr / len(ValidationLabels)))
        ErrorValidationSet.append(Err)

fig, ((ax1, ax2)) = plt.subplots(2, 1)
ax1.plot(ValidationEpoch, AccuracyValidationSet, 'b', label='Ensemble de validation'); ax1.plot(AccuracyTrainingSet, 'r', label="Ensemble d'entrainement");
ax1.set_title("Evolution de la précision"); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Précision (%)')
ax2.plot(ErrorTrainingSet, 'r', label="Ensemble d'entrainement"); ax2.plot(ValidationEpoch, ErrorValidationSet, 'b', label='Ensemble de validation');
ax2.set_title("Evolution de l'erreur à minimiser"); ax2.set_xlabel('Epoch'), ax2.set_ylabel('Erreur')
plt.show()

print(sum(p.numel() for p in N.parameters() if p.requires_grad))