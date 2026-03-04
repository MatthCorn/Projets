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

    def forward(self, x, stride=0):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, stride:x.size(1)+stride]
        return self.dropout(x)


class Rotary(torch.nn.Module):
    def __init__(self, dim, len_seq, len_seq_y=1, base=10000):
        super().__init__()
        self.dim = dim
        self.len_seq = len_seq
        self.len_seq_y = len_seq_y
        self.make_cached(base)

    def forward(self, x):
        # x.shape = (batch_size, len_seq, dim)
        len_seq = self.len_seq * self.len_seq_y
        print("peut poser problème avec l'implémentation MHSA.forward où past_kv != None")
        # rotate_half
        rotate_half_x = (x.reshape(-1, len_seq, self.dim//2, 2) * self.rotator)[..., [1, 0]].reshape(-1, len_seq, self.dim)

        return (x * self.cos_cached) + (rotate_half_x * self.sin_cached)

    def make_cached(self, base):
        if self.len_seq_y == 1:
            pulse = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            t = torch.arange(self.len_seq).type_as(pulse)
            emb = torch.einsum("i,j->ij", t, pulse)
            # emb.shape = (len_seq * len_seq_y, dim//2)
        else:
            pulse = 1.0 / (base ** (torch.arange(0, self.dim // 2, 2).float() / self.dim // 2))
            t_x = torch.arange(self.len_seq).type_as(pulse)
            t_y = torch.arange(self.len_seq_y).type_as(pulse)
            emb_x = torch.einsum("i,j->ij", t_x, pulse)
            # emb_x.shape = (len_seq, dim//4)
            emb_x = emb_x.expand(self.len_seq_y, self.len_seq, self.dim // 4).transpose(0, 1).reshape(self.len_seq * self.len_seq_y, self.dim // 4)
            # emb_x.shape = (len_seq * len_seq_y, dim//4)
            emb_y = torch.einsum("i,j->ij", t_y, pulse)
            # emb_y.shape = (len_seq_y, dim//2)
            emb_y = emb_y.expand(self.len_seq, self.len_seq_y, self.dim // 4).reshape(self.len_seq * self.len_seq_y, self.dim // 4)
            # emb_y.shape = (len_seq * len_seq_y, dim//4)
            emb = torch.cat((emb_x, emb_y), dim=-1)
            # emb.shape = (len_seq * len_seq_y, dim//2)
        emb = emb.unsqueeze(-1).expand(self.len_seq * self.len_seq_y, -1, 2).reshape(self.len_seq * self.len_seq_y, -1)
        # emb.shape = (len_seq * len_seq_y, dim)
        cos_cached = emb.cos()[None, :, :]
        sin_cached = emb.sin()[None, :, :]
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)
        self.register_buffer('rotator', torch.tensor([[[[-1, 1]]]]))


class StandardRotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 70, base: float = 300.0):
        super().__init__()
        self.dim = dim

        # 1. Calcul des fréquences inverses
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        # 2. Création de l'axe temporel (jusqu'à max_seq_len)
        t = torch.arange(max_seq_len, dtype=torch.float32)

        # 3. Produit tensoriel pour obtenir les angles (theta)
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # 4. Concaténation pour correspondre à la dimension totale
        # Ex: si freqs = [f1, f2], emb = [f1, f2, f1, f2] (Style LLaMA)
        emb = torch.cat((freqs, freqs), dim=-1)

        # 5. Mise en cache de cos et sin (ajout d'une dimension batch au début)
        self.register_buffer("cos_cached", emb.cos()[None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, :, :])

    def rotate_half(self, x):
        """Tourne la moitié des dimensions (Standard LLaMA/HuggingFace)"""
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_offset=None):
        """
        Arguments:
            x: Tensor de forme [batch_size, seq_len, dim] ou [batch_size, num_heads, seq_len, head_dim]
            seq_offset: Utilisé quand past_kv != None pour décaler la position (ex: taille du cache)
        """
        if seq_offset is None:
            seq_offset = x.shape[-2]
        seq_len = x.shape[-2]  # Fonctionne que x soit [B, L, D] ou [B, H, L, D]

        # On découpe dynamiquement le cache pour gérer les longueurs variables (KV Cache safe)
        cos = self.cos_cached[:, seq_offset - seq_len: seq_offset, :]
        sin = self.sin_cached[:, seq_offset - seq_len: seq_offset, :]

        # Application de la formule du RoPE
        return (x * cos) + (self.rotate_half(x) * sin)