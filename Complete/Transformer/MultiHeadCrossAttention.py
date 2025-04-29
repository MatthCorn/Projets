import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# Multi-Head Cross Attention
class MHCA(nn.Module):
    def __init__(self, d_att, n_heads, dropout=0.):
        super().__init__()
        d_head, remainder = divmod(d_att, n_heads)
        if remainder:
            raise ValueError("incompatible `d_att` and `n_heads`")

        self.d_att = d_att
        self.d_head = d_head
        self.n_heads = n_heads
        self.key = nn.Linear(d_att, d_att, bias=False)
        self.value = nn.Linear(d_att, d_att, bias=False)
        self.query = nn.Linear(d_att, d_att, bias=False)
        self.linear = nn.Linear(d_att, d_att)
        self.dropout = nn.Dropout(dropout)

        self.ResetParam()

    def forward(self, x_target, x_source, RoPE_Q=lambda u: u, RoPE_K=lambda u: u):
        # x_source.shape = (batch_size, len_source, d_input)
        batch_size, len_source, _ = x_source.shape
        # x_target.shape = (batch_size, len_target, d_latent)
        # OR
        # x_target.shape = (1, len_target, d_latent)
        _, len_target, d_latent = x_target.shape
        Kt = RoPE_K(self.key(x_source)).reshape(batch_size, len_source, self.n_heads, -1).permute(0, 2, 3, 1)
        # Kt.shape = (batch_size, n_heads, d_head, len_source)
        V = self.value(x_source).reshape(batch_size, len_source, self.n_heads, -1).transpose(1, 2)
        # V.shape = (batch_size, n_heads, len_source, d_head)
        Q = RoPE_Q(self.query(x_target)).reshape(-1, len_target, self.n_heads, self.d_head).transpose(1, 2)
        # Q.shape = (batch_size, n_heads, len_target, d_head)
        # OR
        # Q.shape = (1, n_heads, len_target, d_head)

        CA = self.CrossAttention(Q, Kt, V)
        # CA.shape = (batch_size, n_heads, len_target, d_head)

        concat = CA.transpose(1, 2).reshape(batch_size, len_target, -1)
        # Concat.shape = (batch_size, len_target, d_att)

        return self.dropout(self.linear(concat))

    def CrossAttention(self, Q, Kt, V):
        d_head = self.d_head
        # Q.shape = (batch_size, n_heads, len_target, d_head)
        # or
        # Q.shape = (1, n_heads, len_target, d_head)

        # Kt.shape = (batch_size, n_heads, d_head, len_source)

        Q_Kt = torch.matmul(Q, Kt)
        # QKt.shape = (batch_size, n_heads, len_target, len_source)

        attention = Q_Kt / math.sqrt(d_head)
        # Attention.shape = (batch_size, n_heads, len_target, len_source)

        attention = F.softmax(attention, dim=-1)
        out = torch.matmul(attention, V)
        # out.shape = (batch_size, n_heads, len_target, d_head)
        return out

    def ResetParam(self):
        nn.init.xavier_uniform_(self.key.weight)
        self.key.weight.data /= math.sqrt(self.d_att)
        nn.init.xavier_uniform_(self.value.weight)
        self.value.weight.data /= math.sqrt(self.d_att)
        nn.init.xavier_uniform_(self.query.weight)
        self.query.weight.data /= math.sqrt(self.d_att)
