import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_att: int, dropout: float = 0., max_len: int = 5000, device=torch.device('cpu')):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_att, 2) * (-math.log(10000.0) / d_att))
        pe = torch.zeros(1, max_len, d_att)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.to(device=device))

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Rotary(torch.nn.Module):
    def __init__(self, dim, seq_len, base=10000):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.make_cached(base)

    def forward(self, x):
        # x.shape = (batch_size, len_seq, dim)
        batch_size, seq_len, dim = x.shape

        # rotate_half
        rotate_half_x = (x.reshape(batch_size, seq_len, dim//2, 2) * self.rotator)[..., [1, 0]].reshape(batch_size, seq_len, dim)

        return (x * self.cos_cached) + (rotate_half_x * self.sin_cached)

    def make_cached(self, base):
        pulse = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(self.seq_len).type_as(pulse)
        emb = torch.einsum("i,j->ij", t, pulse)
        emb = emb.unsqueeze(-1).expand(self.seq_len, -1, 2).reshape(self.seq_len, -1)
        cos_cached = emb.cos()[None, :, :]
        sin_cached = emb.sin()[None, :, :]
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)
        self.register_buffer('rotator', torch.tensor([[[[-1, 1]]]]))
