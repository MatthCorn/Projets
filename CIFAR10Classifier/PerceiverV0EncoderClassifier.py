from Perceiver.EncoderPerceiver import EncoderLayer
from Transformer.EasyFeedForward import FeedForward
from CIFAR10Classifier.Config import config, MakeLabelSet
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import os

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')

LocalConfig = config(config=2)
LocalConfig.AddParam(d_latent=10, d_att=10, num_heads=1, latent_len=64, max_len=32, d_out=10, normalized=True)

class ClassifierPerceiver(nn.Module):
    def __init__(self, num_enc=2, d_latent=LocalConfig.d_latent, d_input=LocalConfig.d_input, d_att=LocalConfig.d_att,
                 num_heads=LocalConfig.num_heads, latent_len=LocalConfig.latent_len, relative=True):
        super().__init__()
        if relative:
            self.register_buffer("xLatentInit", torch.zeros(1, latent_len, d_latent))
        else :
            self.xLatentInit = nn.parameter.Parameter(torch.normal(mean=torch.zeros(1, latent_len, d_latent)))
        self.encoders = nn.ModuleList()
        for i in range(num_enc):
            self.encoders.append(EncoderLayer(d_latent=d_latent, d_input=d_input, d_att=d_att, num_heads=num_heads, latent_len=latent_len, relative=relative))
        self.FinalClassifier = FeedForward(latent_len*d_latent, 10, widths=[256, 64, 32], dropout=0.05)
        # self.FinalClassifier = FeedForward(d_in=d_latent, d_out=10, widths=[16], dropout=0.05)

    def forward(self, x_input):
        x_latent = self.xLatentInit
        # x_latent.shape = (1, latent_len, d_latent)
        # x_input.shape = (batch_size, input_len, d_input)
        for encoder in self.encoders:
            x_latent = encoder(x_input=x_input, x_latent=x_latent)
           # x_latent.shape = (batch_size, latent_len, d_latent)
        batch_size, _, _ = x_latent.shape
        y = x_latent.reshape(batch_size, -1)
        # y.shape = (batch_size, seq_len*16)
        # y = torch.mean(x_latent, dim=1)
        # # y.shape = (batch_size, d_latent)
        y = self.FinalClassifier(y)
        # y.shape = (batch_size, 10)
        return y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
type = torch.float16

N = ClassifierPerceiver(relative=True, num_enc=2).to(device)

MiniBatchs = [list(range(100*k, 100*(k+1))) for k in range(5)]

optimizer = torch.optim.Adam(N.parameters(), weight_decay=1e-4, lr=1e-4)
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
                    # err = loss(out, labels[MiniBatch])
                    err = torch.norm(N(data[MiniBatch]) - MakeLabelSet(labels[MiniBatch]))
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
                    # Err += 6*float(loss(out, labels[MiniBatch]))
                    Err += 6*float(torch.norm(N(data[MiniBatch]) - MakeLabelSet(labels[MiniBatch])))
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