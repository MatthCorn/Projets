import torch
import math
from torch import nn
import torch.nn.functional as F

# Self Attention
class SA(nn.Module):
    def __init__(self, d_att, dropout=0.):
        super().__init__()

        self.d_att = d_att
        self.key = nn.Linear(d_att, d_att, bias=False)
        self.value = nn.Linear(d_att, d_att, bias=False)
        self.query = nn.Linear(d_att, d_att, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x.shape = (batch_size, len_seq, d_att)
        batch_size, len_seq, d_att = x.shape

        Kt = self.key(x).permute(0, 2, 1)
        # Kt.shape = (batch_size, d_att, len_seq)
        V = self.value(x)
        # V.shape = (batch_size, len_seq, d_att)
        Q = self.query(x)
        # Q.shape = (batch_size, len_seq, d_att)

        SA = self.SelfAttention(Q, Kt, V)
        # RSA.shape = (batch_size, len_seq, d_att)

        concat = SA.transpose(1, 2)
        # concat.shape = (batch_size, len_seq, d_att)

        return self.dropout(concat)

    def SelfAttention(self, Q, Kt, V):

        Q_Kt = torch.matmul(Q, Kt)
        # QKt.shape = (batch_size, len_seq, len_seq)

        attention = Q_Kt / math.sqrt(self.d_att)

        # Attention.shape = (batch_size, len_seq, len_seq)
        attention = F.softmax(attention, dim=-1)

        out = torch.matmul(attention, V)
        # out.shape = (batch_size, len_seq, d_att)
        return out
