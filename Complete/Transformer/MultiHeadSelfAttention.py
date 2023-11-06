import torch
import math
from torch import nn
import torch.nn.functional as F

# Multi-Head Self Attention
class MHSA(nn.Module):
    def __init__(self, d_att, n_heads, dropout=0.):
        super().__init__()
        d_head, remainder = divmod(d_att, n_heads)
        if remainder:
            raise ValueError("incompatible `d_model` and `n_heads`")

        self.d_att = d_att
        self.d_head = d_head
        self.n_heads = n_heads
        self.key = nn.Linear(d_att, d_att, bias=False)
        self.value = nn.Linear(d_att, d_att, bias=False)
        self.query = nn.Linear(d_att, d_att, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x.shape = (batch_size, len_seq, d_model)
        batch_size, len_seq, _ = x.shape
        Kt = self.key(x).reshape(batch_size, len_seq, self.n_heads, self.d_head).permute(0, 2, 3, 1)
        # Kt.shape = (batch_size, n_heads, d_head, len_seq)
        V = self.value(x).reshape(batch_size, len_seq, self.n_heads, self.d_head).transpose(1, 2)
        # V.shape = (batch_size, n_heads, len_seq, d_head)
        Q = self.query(x).reshape(batch_size, len_seq, self.n_heads, self.d_head).transpose(1, 2)
        # Q.shape = (batch_size, n_heads, len_seq, d_head)

        if mask is not None:
            mask = mask[:, :, :len_seq, :len_seq]
            # mask.shape = (1, 1, len_seq, len_seq)

        SA = self.SelfAttention(Q, Kt, V, mask=mask)
        # RSA.shape = (batch_size, n_heads, len_seq, d_head)

        concat = SA.transpose(1, 2).reshape(batch_size, len_seq, self.d_att)
        # concat.shape = (batch_size, len_seq, d_att)

        return self.dropout(concat)

    def SelfAttention(self, Q, Kt, V, mask):
        d_head = self.d_head

        Q_Kt = torch.matmul(Q, Kt)
        # QKt.shape = (batch_size, n_heads, len_seq, len_seq)

        attention = Q_Kt / math.sqrt(d_head)

        if mask is not None:
            # Mask.shape = (1, 1, len_seq, len_seq)
            attention = attention.masked_fill(mask == 0, float("-inf"))

        # Attention.shape = (batch_size, n_heads, len_seq, len_seq)
        attention = F.softmax(attention, dim=-1)

        out = torch.matmul(attention, V)
        # out.shape = (batch_size, n_heads, len_seq, d_head)
        return out