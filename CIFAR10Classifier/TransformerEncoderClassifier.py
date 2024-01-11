from Transformer.EncoderTransformer import EncoderLayer
from Transformer.EasyFeedForward import FeedForward
from CIFAR10Classifier.Config import config, MakeLabelSet
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'OneDrive', 'Documents', 'Python', 'Projets')
# local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'Documents', 'Python', 'Projets')

LocalConfig = config(config=3)
LocalConfig.AddParam(d_att=LocalConfig.d_input, max_len=100, d_out=10, num_heads=4, normalized=True)

class ClassifierTransformer(nn.Module):
    def __init__(self, num_enc=2, d_model=LocalConfig.d_input, num_heads=LocalConfig.num_heads, seq_len=LocalConfig.input_len):
        super().__init__()
        self.Encoders = nn.ModuleList()
        for i in range(num_enc):
            self.Encoders.append(EncoderLayer(d_model=d_model, d_att=d_model, num_heads=num_heads, WidthsFeedForward=[100, 100], max_len=seq_len, MHADropout=0.1, FFDropout=0.05, masked=False))
        self.DimDownScaler = FeedForward(d_model, 32, widths=[64], dropout=0.05)
        self.FinalClassifier = FeedForward(seq_len*32, 10, widths=[256, 64, 32], dropout=0.05)

    def forward(self, x):
        # x.shape = (batch_size, seq_len, d_model)
        y = x
        for layer in self.Encoders:
            y = layer(y)
            # y.shape = (batch_size, seq_len, d_model)
        y = self.DimDownScaler(y)
        # y.shape = (batch_size, seq_len, 32)
        batch_size, _, _ = y.shape
        y = y.reshape(batch_size, -1)
        # y.shape = (batch_size, seq_len*32)
        y = self.FinalClassifier(y)
        # y.shape = (batch_size, 10)
        return y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
type = torch.float16

N = ClassifierTransformer(num_enc=2).to(device)

MiniBatchs = [list(range(100*k, 100*(k+1))) for k in range(5)]

optimizer = torch.optim.Adam(N.parameters(), weight_decay=1e-6, lr=3e-4)
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