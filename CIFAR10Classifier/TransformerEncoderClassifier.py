from Transformer.EncoderTransformer import EncoderLayer
from Transformer.EasyFeedForward import FeedForward
from CIFAR10Classifier.Config import config, MakeLabelSet
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

# local = r'C:\Users\matth\OneDrive\Documents\Python\Projets'
local = r'C:\Users\Matthieu\Documents\Python\Projets'

LocalConfig = config(config=1)
LocalConfig.AddParam(d_att=LocalConfig.d_input, max_len=64, d_out=10)

class ClassifierTransformer(nn.Module):
    def __init__(self, d_model=LocalConfig.d_input, num_heads=LocalConfig.num_heads, seq_len=LocalConfig.input_len):
        super().__init__()
        self.FirstEncoder = EncoderLayer(d_model=d_model, d_att=d_model, num_heads=num_heads, WidthsFeedForward=[100, 100], max_len=seq_len, MHADropout=0.1, FFDropout=0.05, masked=False)
        self.SecondEncoder = EncoderLayer(d_model=d_model, d_att=d_model, num_heads=num_heads, WidthsFeedForward=[100, 100], max_len=seq_len, MHADropout=0.1, FFDropout=0.05, masked=False)
        self.ThirdEncoder = EncoderLayer(d_model=d_model, d_att=d_model, num_heads=num_heads, WidthsFeedForward=[100, 100], max_len=seq_len, MHADropout=0.1, FFDropout=0.05, masked=False)
        self.DimDownScaler = FeedForward(d_model, 32, widths=[64], dropout=0.05)
        self.FinalClassifier = FeedForward(seq_len*32, 10, widths=[256, 64, 32], dropout=0.05)

    def forward(self, x):
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
type = torch.float16

N = ClassifierTransformer().to(device)

MiniBatchs = [list(range(100*k, 100*(k+1))) for k in range(5)]

optimizer = torch.optim.Adam(N.parameters(), weight_decay=1e-5)
scaler = GradScaler()
loss = nn.CrossEntropyLoss()

ErrorTrainingSet = []
AccuracyValidationSet = []
ValidationImageSet, ValidationLabels = LocalConfig.LoadValidation(local)

LittleBatchs = [list(range(500*k, 500*(k+1))) for k in range(20)]

for i in tqdm(range(1)):
    CurrentError = 0
    for j in range(1, 6):
        BatchData, BatchLabels = LocalConfig.LoadBatch(j, local)
        for LittleBatch in LittleBatchs:
            data, labels = BatchData[LittleBatch].to(device), BatchLabels[LittleBatch].to(device)
            for MiniBatch in MiniBatchs:
                with torch.autocast(device_type='cuda', dtype=type):
                    out = N(data[MiniBatch])
                    err = loss(out, labels[MiniBatch])
                    # err = torch.norm(N(data[MiniBatch]) - MakeLabelSet(labels[MiniBatch]))
                scaler.scale(err).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                CurrentError += float(err)
    ErrorTrainingSet.append(CurrentError)
    if i % 5 == 0:
        Err = 0
        for LittleBatch in LittleBatchs:
            data, labels = ValidationImageSet[LittleBatch].to(device), ValidationLabels[LittleBatch].to(device)
            with torch.autocast(device_type='cuda', dtype=type):
                Err += torch.count_nonzero(torch.argmax(N(data), dim=1) - labels)
        AccuracyValidationSet.append(float(1 - Err / len(ValidationLabels)))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(AccuracyValidationSet); ax1.set_title("Précision sur l'ensemble de validation")
ax2.plot(ErrorTrainingSet); ax2.set_title("Erreur sur l'ensemble de test")
ax3.plot(torch.norm(N.FirstEncoder.MultiHeadAttention.Er, dim=1).cpu().detach().numpy()); ax3.set_title("Norme de Er sur le premier encodeur")
ax4.plot(torch.norm(N.SecondEncoder.MultiHeadAttention.Er, dim=1).cpu().detach().numpy()); ax4.set_title("Norme de Er sur le second encodeur")
plt.show()

print(sum(p.numel() for p in N.parameters() if p.requires_grad))