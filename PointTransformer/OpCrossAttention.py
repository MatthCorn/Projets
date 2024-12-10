import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from PointTransformer.CustomReLU import CustomReLU

# Cross Attention
class CA(nn.Module):
    def __init__(self, d_att, d_group=1, dropout=0.):
        super().__init__()

        self.d_att = d_att
        self.d_group = d_group
        n_group, remainder = divmod(d_att, d_group)
        if remainder:
            raise ValueError("incompatible `d_att` and `d_group`")
        self.n_group = n_group
        self.key = nn.Linear(d_att, d_att, bias=False)
        self.value = nn.Linear(d_att, d_att, bias=False)
        self.query = nn.Linear(d_att, d_att, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.group_MLP = nn.Sequential(CustomReLU(), nn.Linear(d_group, 1), nn.ReLU())

        self.ResetParam()

    def forward(self, x_source, x_target, positional_adding_bias_ts=0, positional_multiplying_bias_ts=1):
        # x_target.shape = (batch_size, len_seq_1, d_att)
        # x_source.shape = (batch_size, len_seq_2, d_att)
        batch_size, len_seq_1, d_att = x_target.shape

        K = self.key(x_source)
        # K.shape = (batch_size, len_seq_2, d_att)
        V = self.value(x_source)
        # V.shape = (batch_size, len_seq_2, d_att)
        Q = self.query(x_target)
        # Q.shape = (batch_size, len_seq_1, d_att)

        GVA = self.GroupedVectorAttention(Q, K, V, positional_adding_bias_ts, positional_multiplying_bias_ts)
        # GVA.shape = (batch_size, len_seq_1, d_group, n_group)

        concat = GVA.reshape(batch_size, len_seq_1, d_att)
        # concat.shape = (batch_size, len_seq_1, d_att)

        return self.dropout(concat)

    def GroupedVectorAttention(self, Q, K, V, positional_adding_bias, positional_multiplying_bias):
        d_group = self.d_group
        n_group = self.n_group
        batch_size, len_seq_1, d_att = Q.shape
        _, len_seq_2, _ = K.shape

        Q_grouped = Q.view(batch_size, len_seq_1, 1, n_group, d_group)
        K_grouped = K.view(batch_size, 1, len_seq_2, n_group, d_group)
        V_grouped = V.view(batch_size, len_seq_2, d_group, n_group)

        diff_Q_K = Q_grouped - K_grouped
        # diff_Q_K.shape = (batch_size, len_seq_1, len_seq_2, n_group, d_group)

        attention = self.group_MLP(diff_Q_K).squeeze(-1)
        # attention.shape = (batch_size, len_seq_1, len_seq_2, n_group)

        attention = attention * positional_multiplying_bias + positional_adding_bias

        attention = F.softmax(attention, dim=2)

        out = torch.einsum('ijbl,ibkl->ijkl', attention, V_grouped)

        out = out.reshape(batch_size, len_seq_1, d_att)

        return out

    def ResetParam(self):
        nn.init.xavier_uniform_(self.key.weight)
        self.key.weight.data /= math.sqrt(self.d_att)
        nn.init.xavier_uniform_(self.value.weight)
        self.value.weight.data /= math.sqrt(self.d_att)
        nn.init.xavier_uniform_(self.query.weight)
        self.query.weight.data /= math.sqrt(self.d_att)