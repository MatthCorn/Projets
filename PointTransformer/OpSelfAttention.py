import torch
import math
from torch import nn
import torch.nn.functional as F
from PointTransformer.CustomReLU import CustomReLU

# Self Attention
class SA(nn.Module):
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

    def forward(self, x, positional_adding_bias=0, positional_multiplying_bias=1, mask=None):
        # x.shape = (batch_size, len_seq, d_att)
        batch_size, len_seq, d_att = x.shape

        K = self.key(x)
        # K.shape = (batch_size, len_seq, d_att)
        V = self.value(x)
        # V.shape = (batch_size, len_seq, d_att)
        Q = self.query(x)
        # Q.shape = (batch_size, len_seq, d_att)

        if mask is not None:
            mask = mask[:len_seq, :len_seq]
            # mask.shape = (len_seq, len_seq, 1)

        GVA = self.GroupedVectorAttention(Q, K, V, positional_adding_bias, positional_multiplying_bias, mask=mask)
        # GVA.shape = (batch_size, len_seq, d_att)

        return self.dropout(GVA)

    def GroupedVectorAttention(self, Q, K, V, positional_adding_bias, positional_multiplying_bias, mask=None):
        d_group = self.d_group
        n_group = self.n_group
        batch_size, len_seq, d_att = Q.shape

        Q_grouped = Q.view(batch_size, len_seq, 1, n_group, d_group)
        K_grouped = K.view(batch_size, 1, len_seq, n_group, d_group)
        V_grouped = V.view(batch_size, len_seq, d_group, n_group)

        diff_Q_K = Q_grouped - K_grouped
        # diff_Q_K.shape = (batch_size, len_seq, len_seq, n_group, d_group)

        attention = self.group_MLP(diff_Q_K).squeeze(-1)
        # attention.shape = (batch_size, len_seq, len_seq, n_group)

        attention = attention * positional_multiplying_bias + positional_adding_bias

        if mask is not None:
            # Mask.shape = (len_seq, len_seq, 1)
            attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(attention, dim=2)

        out = torch.einsum('ijbl,ibkl->ijkl', attention, V_grouped)

        out = out.reshape(batch_size, len_seq, d_att)

        return out

    def ResetParam(self):
        nn.init.xavier_uniform_(self.key.weight)
        self.key.weight.data /= math.sqrt(self.d_att)
        nn.init.xavier_uniform_(self.value.weight)
        self.value.weight.data /= math.sqrt(self.d_att)
        nn.init.xavier_uniform_(self.query.weight)
        self.query.weight.data /= math.sqrt(self.d_att)