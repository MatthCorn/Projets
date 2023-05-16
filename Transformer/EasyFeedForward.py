from torch import nn
import torch.nn.functional as F

# Customizable Feed-Forward Network in terms of number of layer and their widths
class FeedForward(nn.Module):
    def __init__(self, d_in, d_out, widths = [512], dropout = 0.1):
        super().__init__()
        # self.linears lists all linear layers of the network
        self.linears = []
        self.linears.append(nn.Linear(d_in, widths[0]))
        # Need to add manually the layers as module of the main network as there are not directly its child from the object programing POV
        self.add_module('linear_in', self.linears[-1])
        for i in range(1,len(widths)):
            self.linears.append(nn.Linear(widths[i-1], widths[i]))
            # Need to add manually the layers as module of the main network as there are not directly its child from the object programing POV
            self.add_module('linear_'+str(i), self.linears[-1])
        self.linears.append(nn.Linear(widths[-1], d_out))
        self.add_module('linear_out', self.linears[-1])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = self.dropout(F.relu(linear(x)))
        x = self.linears[-1](x)
        return x
