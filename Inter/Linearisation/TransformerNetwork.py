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

    def forward(self, input, mask, RoPE=lambda u: u, past_kv=None):
        if self.norm == 'pre':
            y = self.mask_self_attention(self.first_layer_norm(input), mask=mask, RoPE=RoPE, past_kv=past_kv) + input
            y = self.feed_forward(self.third_layer_norm(y)) + y
        elif self.norm == 'post':
            y = self.first_layer_norm(self.mask_self_attention(input, mask=mask, RoPE=RoPE, past_kv=past_kv) + input)
            y = self.third_layer_norm(self.feed_forward(y) + y)
        else:
            y = self.mask_self_attention(input, mask=mask, RoPE=RoPE, past_kv=past_kv) + input
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
        self.n_heads = n_head
        self.d_head = hidden_dim // n_head
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
        self.past_kv = None


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
            x = layer(x, self.mask)

        # C. Sortie directe (Pas d'attention, pas de combinaison complexe)
        pred = self.head(x)

        # Calcul optionnel de distance (votre logique existante)
        is_next = self.next_detector(x, self.next_token())

        return pred, is_next

    def step(self, input_1, input_2, next_mask=None):
        """
        Mode Inférence Pas-à-Pas (Bufferisé).
        buffer: contient l'historique brut des embeddings [Batch, Hidden, T_history]
        """
        # 1. Gestion du Buffer
        if self.past_kv is None:
            batch_size, *_ = input_1.shape
            self.past_kv = [[torch.zeros(batch_size, 0, self.n_heads, self.d_head),
                             torch.zeros(batch_size, self.n_heads, 0, self.d_head)].copy()
                            for _ in range(len(self.transformer_core))]

        # 2. Embeddings de l'instant t
        if next_mask is None:
            next_mask = torch.zeros(*input_1.shape[:-1], 1, 1).to(input_1.device)
        else:
            next_mask = next_mask.unsqueeze(-1)

        x_1 = self.embedding_1(input_1.unsqueeze(1))
        x_2 = self.embedding_2(input_2.unsqueeze(1))
        x_2 = x_2 * (1 - next_mask) + next_mask * self.next_token()

        # x : [Batch, 1, Hidden]
        x = self.merge_embeddings(torch.cat([x_1, x_2], dim=-1))
        x = self.pos_encoding(x, stride=self.past_kv[0][0].shape[1])

        for i, layer in enumerate(self.transformer_core):
            x = layer(x, self.mask, past_kv=self.past_kv[i])

        # C. Sortie directe (Pas d'attention, pas de combinaison complexe)
        pred = self.head(x)

        # Calcul optionnel de distance (votre logique existante)
        is_next = self.next_detector(x, self.next_token())

        return pred, is_next

if __name__ == '__main__':
    N = Transformer(
        10,
        10,
        64,
        5,
        len_out=20,
        n_layer=5,
        n_head=4,
        dropout=0.0
    )
    x1 = torch.normal(0, 1, (40, 20, 10))
    x2 = torch.normal(0, 1, (40, 20, 10))
    mask = torch.randint(0, 2, (40, 20, 1))
    y1, is_next_1 = N(x1, x2, mask)

    y_list = []
    is_next_list = []
    for k in range(x1.shape[1]):
        x1_k = x1[:, k]
        x2_k = x2[:, k]
        next_mask = mask[:, k]
        y_k, is_next = N.step(x1_k, x2_k, next_mask)
        y_list.append(y_k)
        is_next_list.append(is_next)
    y2 = torch.cat(y_list, dim=1)
    is_next_2 = torch.cat(is_next_list, dim=1)

    print(y2[0, :, 0])
    print(y1[0, :, 0])

    print(is_next_1[0, :, 0])
    print(is_next_2[0, :, 0])