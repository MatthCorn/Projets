import pickle
import torch
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def MakeLabelSet(x):
    out = torch.zeros(len(x), 10)
    for i in range(len(x)):
        out[i, int(x[i])] = 1
    return out

class config():
    def __init__(self, config=0):
        self.config = config
        # La séquence est de la forme c_1, c_2, c_3, ..., c_i, ... où c_i est le vecteur avec la colonne de pixels i rouge, puis la verte, puis la bleue
        if config == 0:
            self.d_input = 96
            self.input_len = 32
        # La séquence est de la forme c_1, c_2, c_3, ..., c_3i, c_3i+1, c_3i+2 ... où c_3i+k est la colonne de pixels i rouge(k=1), verte(k=2) ou bleue(k=2)
        if config == 1:
            self.d_input = 32
            self.input_len = 96
        # La séquence est de la forme c_1, c_2, c_3, ..., c_n*i+j, ... où c_n*i+j est le pixel (i,j) de dim 3 (RGB)
        if config == 2:
            self.d_input = 3
            self.input_len = 1024

    def LoadBatch(self, i, local, device=torch.device('cpu'), type=torch.float32):
        dict = unpickle(local + r'\CIFAR10Classifier\data_batch_' + str(i))
        data = torch.tensor(dict[b'data'])
        # data.shape = (BatchSize,3*n*n)
        BatchSize, temp = data.shape
        n = int((temp / 3) ** 0.5)
        data = data.reshape(BatchSize, 3, n, n)
        data = data.transpose(1, 2)
        if self.config == 2:
            data = data.transpose(2, 3)
        data = data.reshape(BatchSize, self.input_len, self.d_input)
        # data.shape = (batch_size, input_len, d_input)
        return data.to(device, torch.float16), MakeLabelSet(dict[b'labels']).to(device, type)

    def LoadValidation(self, local, device=torch.device('cpu'), type=torch.float32):
        dict = unpickle(local + r'\CIFAR10Classifier\test_batch')
        data = torch.tensor(dict[b'data'])
        # data.shape = (BatchSize,3*n*n)
        BatchSize, temp = data.shape
        n = int((temp / 3) ** 0.5)
        data = data.reshape(BatchSize, 3, n, n)
        data = data.transpose(1, 2)
        if self.config == 2:
            data = data.transpose(2, 3)
        data = data.reshape(BatchSize, self.input_len, self.d_input)
        # data.shape = (batch_size, input_len, d_input)
        return data.to(device, type), torch.tensor(dict[b'labels']).to(device)


    def PlotImage(self, i, data):
        if self.config == 2:
            im = data[i].reshape(32, 32, 3)
        else:
            im = data[i].reshape(32, 3, 32).transpose(1, 2)
        plt.imshow(im.numpy())
        plt.show()

    def AddParam(self, d_latent=32, d_att=32, num_heads=4, latent_len=32, max_len=64, d_out=10):
        self.d_latent = d_latent
        self.d_att = d_att
        self.num_heads = num_heads
        self.latent_len = latent_len
        self.max_len = max_len
        self.d_out = d_out




