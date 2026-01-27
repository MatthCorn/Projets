import torch
import torch.nn as nn
from Complete.Transformer.EasyFeedForward import FeedForward

class Conv(nn.Module):
    """
    Implémentation du bloc convolutif pour l'ENCODEUR (Non-Causal / Bidirectionnel).
    Fidèle à l'article Yi Tay et al. (2022).

    Caractéristiques :
    - Remplace le MHSA dans l'encodeur.
    - Vision globale locale (fenêtre glissante symétrique).
    """

    def __init__(self, d_model, kernel_size=7, dropout=0.1):
        super().__init__()

        # Padding Symétrique ("Same" padding)
        # Pour une kernel_size impaire (ex: 7), on ajoute (K-1)/2 zéros de chaque côté.
        # Ex: K=7 -> padding=3. L'entrée de taille L ressortira avec taille L.
        if kernel_size % 2 == 0:
            raise ValueError("Pour un encodeur avec padding symétrique, kernel_size doit être impair.")

        self.padding = kernel_size // 2

        # 1. Depthwise Convolution (Mélange Temporel)
        # Regarde à gauche ET à droite.
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=self.padding,  # Padding égal à gauche et à droite
            groups=d_model,  # Depthwise
            bias=False
        )

        # 2. Pointwise Convolution (Mélange des Canaux)
        self.pointwise_conv = nn.Linear(d_model, d_model)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, **kwargs):
        # x: [Batch, Time, Hidden]

        # Passage en format Conv: [Batch, Hidden, Time]
        x_in = x.transpose(1, 2)

        # A. Convolution (Non-causale)
        # Grâce au padding symétrique, la sortie a directement la bonne taille L.
        # Pas besoin de couper (chomp) ici.
        out = self.depthwise_conv(x_in)

        # Retour en format [Batch, Time, Hidden]
        out = out.transpose(1, 2)

        # B. Activation & Norm
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)

        # C. Mélange des Canaux
        out = self.pointwise_conv(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, kernel_size, width_FF=[32], dropout_Conv=0., dropout_FF=0., norm='post'):
        super().__init__()
        self.norm = norm
        if self.norm != 'none':
            self.first_layer_norm = nn.LayerNorm(d_model)
            self.second_layer_norm = nn.LayerNorm(d_model)
        self.mixing_module = Conv(d_model, kernel_size=kernel_size, dropout=dropout_Conv)
        self.feed_forward = FeedForward(d_model, d_model, widths=width_FF, dropout=dropout_FF)

    def forward(self, x):
        if self.norm == 'pre':
            y = self.mixing_module(self.first_layer_norm(x)) + x
            y = self.feed_forward(self.second_layer_norm(y) + y)
        elif self.norm == 'post':
            y = self.first_layer_norm(self.mixing_module(x) + x)
            y = self.second_layer_norm(self.feed_forward(y) + y)
        else:
            y = self.mixing_module(x) + x
            y = self.feed_forward(y) + y
        return y