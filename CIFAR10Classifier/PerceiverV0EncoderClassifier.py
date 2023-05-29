from Perceiver.EncoderPerceiver import EncoderLayer
from Transformer.EasyFeedForward import FeedForward
from CIFAR10Classifier.Config import config
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

local = r'C:\Users\matth\OneDrive\Documents\Python\Projets'
# local = r'C:\Users\Matthieu\Documents\Python\Projets'

LocalConfig = config(config=2)
LocalConfig.AddParam(d_latent=32, d_att=32, num_heads=4, latent_len=32, max_len=64, d_out=10)

class ClassifierPerceiver(nn.Module):
    def __init__(self, d_latent=LocalConfig.d_latent, d_input=LocalConfig.d_input, d_att=LocalConfig.d_att,
                 num_heads=LocalConfig.num_heads, latent_len=LocalConfig.latent_len):
        super().__init__()
        # self.xLatentInit = nn.parameter.Parameter(torch.normal(mean=torch.zeros(1, latent_len, d_latent)))
        self.register_buffer("xLatentInit", torch.zeros(1, latent_len, d_latent))
        self.EncoderLayer1 = EncoderLayer(d_latent=d_latent, d_input=d_input, d_att=d_att, num_heads=num_heads, latent_len=latent_len)
        self.EncoderLayer2 = EncoderLayer(d_latent=d_latent, d_input=d_input, d_att=d_att, num_heads=num_heads, latent_len=latent_len)
        self.EncoderLayer3 = EncoderLayer(d_latent=d_latent, d_input=d_input, d_att=d_att, num_heads=num_heads, latent_len=latent_len)
        self.EncoderLayer4 = EncoderLayer(d_latent=d_latent, d_input=d_input, d_att=d_att, num_heads=num_heads, latent_len=latent_len)
        self.EncoderLayer5 = EncoderLayer(d_latent=d_latent, d_input=d_input, d_att=d_att, num_heads=num_heads, latent_len=latent_len)
        self.FinalClassifier = FeedForward(latent_len*d_latent, 10, widths=[256, 64, 32], dropout=0.05)
        # self.FinalClassifier = FeedForward(d_in=d_latent, d_out=10, widths=[16], dropout=0.05)

    def forward(self, x_input):
        x_latent = self.xLatentInit
        # x_latent.shape = (1, latent_len, d_latent)
        # x_input.shape = (batch_size, input_len, d_input)
        x_latent = self.EncoderLayer1(x_input=x_input, x_latent=x_latent)
        # x_latent.shape = (batch_size, latent_len, d_latent)
        x_latent = self.EncoderLayer2(x_input=x_input, x_latent=x_latent)
        # x_latent.shape = (batch_size, latent_len, d_latent)
        x_latent = self.EncoderLayer3(x_input=x_input, x_latent=x_latent)
        x_latent = self.EncoderLayer4(x_input=x_input, x_latent=x_latent)
        x_latent = self.EncoderLayer5(x_input=x_input, x_latent=x_latent)
        batch_size, _, _ = x_latent.shape
        y = x_latent.reshape(batch_size, -1)
        # y.shape = (batch_size, seq_len*16)
        # y = torch.mean(x_latent, dim=1)
        # # y.shape = (batch_size, d_latent)
        y = self.FinalClassifier(y)
        # y.shape = (batch_size, 10)
        return y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = ClassifierPerceiver().to(device)

MiniBatchs = [list(range(100*k, 100*(k+1))) for k in range(5)]

optimizer = torch.optim.Adam(N.parameters(), weight_decay=1e-6)

ErrorTrainingSet = []
AccuracyValidationSet = []
ValidationImageSet, ValidationLabels = LocalConfig.LoadValidation(local)

LittleBatchs = [list(range(500*k, 500*(k+1))) for k in range(20)]

for i in range(100):
    print('i = ' + str(i))
    CurrentError = 0
    for j in range(1,6):
        BatchData, BatchLabels = LocalConfig.LoadBatch(j, local)
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
