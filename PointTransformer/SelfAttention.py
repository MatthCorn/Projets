import torch
import math
from torch import nn
import torch.nn.functional as F

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

        self.group_MLP = nn.Sequential()
        if (not d_att % d_group) and (d_att != d_group):
            d = d_group
            self.group_MLP.append(nn.Linear(d, d))
            self.group_MLP.append(nn.ReLU())
            while d > 16:
                self.group_MLP.append(nn.Linear(d, int(d/4)))
                self.group_MLP.append(nn.ReLU())
                d = int(d/4)
            self.group_MLP.append(nn.Linear(d, 1))
            self.group_MLP.append(nn.ReLU())

        self.ResetParam()

    def forward(self, x, positional_adding_bias, positional_multiplying_bias, mask=None):
        # x.shape = (batch_size, len_seq, d_att)
        batch_size, len_seq, d_att = x.shape

        K = self.key(x)
        # K.shape = (batch_size, len_seq, d_att)
        V = self.value(x)
        # V.shape = (batch_size, len_seq, d_att)
        Q = self.query(x)
        # Q.shape = (batch_size, len_seq, d_att)

        if mask is not None:
            mask = mask[:, :, :, :len_seq, :len_seq]
            # mask.shape = (1, 1, 1, len_seq, len_seq)

        GVA = self.GroupedVectorAttention(Q, K, V, positional_adding_bias, positional_multiplying_bias, mask=mask)
        # GVA.shape = (batch_size, len_seq, d_group, n_group)

        concat = GVA.reshape(batch_size, len_seq, d_att)
        # concat.shape = (batch_size, len_seq, d_att)

        return self.dropout(concat)

    def GroupedVectorAttention(self, Q, K, V, positional_adding_bias, positional_multiplying_bias, mask):
        d_group = self.d_group
        n_group = self.n_group

        # K.shape = (batch_size, len_seq, d_att)
        # V.shape = (batch_size, len_seq, d_att)
        # Q.shape = (batch_size, len_seq, d_att)

        diff_Q_K = Q.transpose(1, 2).unsqueeze(-1) - K.transpose(1, 2).unsqueeze(-2)
        (batch_size, d_att, len_seq, _) = diff_Q_K.shape
        diff_Q_K = diff_Q_K.permute(0, 2, 3, 1)
        diff_Q_K = diff_Q_K.reshape(batch_size, len_seq, len_seq, n_group, d_group)

        attention = self.group_MLP(diff_Q_K)
        # attention.shape = (batch_size, len_seq, len_seq, n_group, 1)
        attention = attention.permute(0, 3, 4, 1, 2)
        # attention.shape = (batch_size, n_group, 1, len_seq, len_seq)
        attention = attention * positional_multiplying_bias + positional_adding_bias
        if mask is not None:
            # Mask.shape = (1, 1, 1, len_seq, len_seq)
            attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        attention = attention.transpose(1, 3)
        # attention.shape = (batch_size, len_seq, 1, n_group, len_seq)


        V = V.reshape(batch_size, 1, len_seq, d_group, n_group)
        V = V.permute(0, 1, 3, 4, 2)
        # V.shape = (batch_size, 1, d_group, n_group, len_seq)

        out = torch.matmul(attention.unsqueeze(-2), V.unsqueeze(-1))
        out = out.reshape(batch_size, len_seq, d_att)

        return out

    def ResetParam(self):
        nn.init.xavier_uniform_(self.key.weight)
        self.key.weight.data /= math.sqrt(self.d_att)
        nn.init.xavier_uniform_(self.value.weight)
        self.value.weight.data /= math.sqrt(self.d_att)
        nn.init.xavier_uniform_(self.query.weight)
        self.query.weight.data /= math.sqrt(self.d_att)

if __name__ == '__main__':
    Att = SA(64, 16, None)
    input = torch.normal(0, 1, (100, 10, 64))
    output = Att(input)