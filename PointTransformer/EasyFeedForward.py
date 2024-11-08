from torch import nn
import torch.nn.functional as F

# Customizable Feed-Forward Network in terms of number of layer and their widths
class FeedForward(nn.Module):
    def __init__(self, d_in, d_out, widths=[32], dropout=0.):
        super().__init__()
        # self.linears lists all linear layers of the network
        self.linears = nn.ModuleList()

        if widths == []:
            self.linears.append(nn.Linear(d_in, d_out))

        else:
            self.linears.append(nn.Linear(d_in, widths[0]))
            for i in range(1, len(widths)):
                self.linears.append(nn.Linear(widths[i-1], widths[i]))
            self.linears.append(nn.Linear(widths[-1], d_out))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = self.dropout(F.relu(linear(x)))
        x = self.linears[-1](x)
        return x
