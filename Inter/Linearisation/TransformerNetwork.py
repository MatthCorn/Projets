import torch
import torch.nn as nn
from Complete.Transformer.MultiHeadSelfAttention import MHSA
from Complete.Transformer.EasyFeedForward import FeedForward
from Complete.Transformer.LearnableModule import LearnableParameters
from Complete.Transformer.PositionalEncoding import PositionalEncoding
from Tools.MCCutils import CosineDetector

class CausalEncoderLayer(nn.Module):
    def __init__(self, d_att, n_heads, width_FF=[32], dropout_A=0., dropout_FF=0., norm='post'):
        super().__init__()
        self.norm = norm
        if self.norm != 'none':
            self.first_layer_norm = nn.LayerNorm(d_att)
            self.second_layer_norm = nn.LayerNorm(d_att)
            self.third_layer_norm = nn.LayerNorm(d_att)

        self.mask_self_attention = MHSA(d_att, n_heads, dropout=dropout_A)
        self.feed_forward = FeedForward(d_att, d_att, widths=width_FF, dropout=dropout_FF)

    def forward(self, input, mask, RoPE=lambda u: u):
        if self.norm == 'pre':
            y = self.self_attention(self.first_layer_norm(input), mask=mask, RoPE=RoPE) + input
            y = self.feed_forward(self.third_layer_norm(y)) + y
        elif self.norm == 'post':
            y = self.first_layer_norm(self.self_attention(input, mask=mask, RoPE=RoPE) + input)
            y = self.third_layer_norm(self.feed_forward(y) + y)
        else:
            y = self.self_attention(input, mask=mask, RoPE=RoPE) + input
            y = self.feed_forward(y) + y
        return y


class Transformer(nn.Module):
    def __init__(
            self,
            input_dim_1,
            input_dim_2,
            hidden_dim,
            output_dim,
            use_mlp_head=True,
            len_out=100,
            n_layer=5,
            n_head=4,
            dropout=0.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_encoding = PositionalEncoding(d_att=hidden_dim, dropout=dropout, max_len=len_out)

        # --- 1. EMBEDDINGS (Votre logique spécifique) ---
        self.embedding_1 = nn.Sequential(
            nn.Linear(input_dim_1, hidden_dim), nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embedding_2 = nn.Sequential(
            nn.Linear(input_dim_2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.merge_embeddings = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        self.next_token = LearnableParameters(torch.normal(0, 1, [1, 1, hidden_dim]))

        # --- 2. LE CŒUR du réseau (Remplace LSTM + Attention) ---
        self.transformer_core = nn.ModuleList([CausalEncoderLayer(hidden_dim, n_head, dropout_A=dropout, dropout_FF=dropout) for _ in range(n_layer)])

        self.next_detector = CosineDetector()

        # --- 3. SORTIE ---
        # On projette la sortie (qui a pour dim le dernier canal de tcn_channels)
        if use_mlp_head:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.head = nn.Linear(hidden_dim, output_dim)

        self.register_buffer("mask", torch.tril(torch.ones(len_out, len_out)).unsqueeze(0).unsqueeze(0), persistent=False)


    def forward(self, input_1, input_2, next_mask=None):
        """Mode Entraînement (Parallèle)"""
        if next_mask is None:
            next_mask = torch.zeros(*input_1.shape[:-1], 1).to(input_1.device)

        # A. Préparation des entrées
        x_1 = self.embedding_1(input_1)
        x_2 = self.embedding_2(input_2)
        x_2 = x_2 * (1 - next_mask) + next_mask * self.next_token()

        # Fusion -> [Batch, Time, Hidden]
        x = self.merge_embeddings(torch.cat([x_1, x_2], dim=-1))
        x = self.pos_encoding(x)

        for layer in self.transformer_core:
            x = layer(x, mask)

        # C. Sortie directe (Pas d'attention, pas de combinaison complexe)
        pred = self.head(x)

        # Calcul optionnel de distance (votre logique existante)
        is_next = self.next_detector(x, self.next_token())

        return pred, is_next

    def step(self, input_1, input_2, buffer=None, next_mask=None):
        """
        Mode Inférence Pas-à-Pas (Bufferisé).
        buffer: contient l'historique brut des embeddings [Batch, Hidden, T_history]
        """
        # 1. Embeddings de l'instant t
        if next_mask is None:
            next_mask = torch.zeros(*input_1.shape[:-1], 1, 1).to(input_1.device)
        else:
            next_mask = next_mask.unsqueeze(-1)

        x_1 = self.embedding_1(input_1.unsqueeze(1))
        x_2 = self.embedding_2(input_2.unsqueeze(1))
        x_2 = x_2 * (1 - next_mask) + next_mask * self.next_token()

        # x_t : [Batch, 1, Hidden]
        x_t = self.merge_embeddings(torch.cat([x_1, x_2], dim=-1))

        # 2. Gestion du Buffer
        if buffer is None:
            buffer = x_t
        else:
            buffer = torch.cat([buffer, x_t], dim=1)

        # 3. Inférence
        # On applique le TCN sur tout le buffer
        tcn_out_seq = self.tcn(buffer)  # [Batch, Channels, Buffer_Len]

        # On ne prend que le DERNIER point temporel (l'instant présent)
        h_t = tcn_out_seq[:, :, -1]  # [Batch, Channels]

        if self.use_layernorm:
            h_t = self.post_tcn_ln(h_t)

        # 4. Sortie
        y_t = self.head(h_t)  # [Batch, Output_Dim]

        # (Pour compatibilité avec votre code existant qui attend 4 sorties)
        # Ici next_dist est calculé sur le vecteur courant
        is_next = self.next_detector(h_t.unsqueeze(1), self.next_token())

        # On retourne le buffer mis à jour au lieu de (hidden, H_past)
        # H_past n'est plus utile car pas d'attention
        return y_t, buffer, is_next

if __name__ == '__main__':
    N = MemoryUpdateTCN(10,
            10,
            64,  # Taille interne des embeddings
            5,
            # C'est ici que vous réglez la puissance du TCN :
            tcn_channels=[64, 64, 64, 64, 64],  # 5 blocs → dilation jusqu'à 16
            kernel_size=3,
            dropout=0.0,
            use_layernorm=True,
    )
    x1 = torch.normal(0, 1, (40, 20, 10))
    x2 = torch.normal(0, 1, (40, 20, 10))
    mask = torch.randint(0, 2, (40, 20, 1))
    y1, is_next_1 = N(x1, x2, mask)

    y_list = []
    is_next_list = []
    buffer = None
    for k in range(x1.shape[1]):
        x1_k = x1[:, k]
        x2_k = x2[:, k]
        next_mask = mask[:, k]
        y_k, buffer, is_next = N.step(x1_k, x2_k, buffer, next_mask)
        y_list.append(y_k.unsqueeze(1))
        is_next_list.append(is_next)
    y2 = torch.cat(y_list, dim=1)
    is_next_2 = torch.cat(is_next_list, dim=1)

    print(y2[0, :, 0])
    print(y1[0, :, 0])

    print(is_next_1[0, :, 0])
    print(is_next_2[0, :, 0])