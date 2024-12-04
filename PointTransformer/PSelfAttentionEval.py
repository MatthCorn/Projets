import torch
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

        self.group_MLP = nn.Sequential(nn.Linear(d_att, 1), nn.ReLU())


    def forward(self, x):
        # x.shape = (batch_size, len_seq, d_att)
        batch_size, len_seq, d_att = x.shape

        K = self.key(x)
        # K.shape = (batch_size, len_seq, d_att)
        V = self.value(x)
        # V.shape = (batch_size, len_seq, d_att)
        Q = self.query(x)
        # Q.shape = (batch_size, len_seq, d_att)

        GVA = self.GroupedVectorAttention(Q, K, V)
        # GVA.shape = (batch_size, len_seq, d_att)

        return self.dropout(GVA)

    def GroupedVectorAttention(self, Q, K, V):

        # K.shape = (batch_size, len_seq, d_att)
        # V.shape = (batch_size, len_seq, d_att)
        # Q.shape = (batch_size, len_seq, d_att)

        diff_Q_K = Q.transpose(1, 2).unsqueeze(-1) - K.transpose(1, 2).unsqueeze(-2)
        (batch_size, d_att, len_seq, _) = diff_Q_K.shape
        diff_Q_K = diff_Q_K.permute(0, 2, 3, 1)
        diff_Q_K = diff_Q_K.reshape(batch_size, len_seq, len_seq, 1, d_att)

        attention = self.group_MLP(diff_Q_K)
        # attention.shape = (batch_size, len_seq, len_seq, 1, 1)
        attention = attention.permute(0, 3, 4, 1, 2)
        # attention.shape = (batch_size, 1, 1, len_seq, len_seq)

        attention = F.softmax(attention, dim=-1)
        attention = attention.transpose(1, 3)
        # attention.shape = (batch_size, len_seq, 1, 1, len_seq)


        V = V.reshape(batch_size, 1, len_seq, d_att, 1)
        V = V.permute(0, 1, 3, 4, 2)
        # V.shape = (batch_size, 1, d_att, 1, len_seq)

        out = torch.matmul(attention.unsqueeze(-2), V.unsqueeze(-1))
        out = out.reshape(batch_size, len_seq, d_att)

        return out
