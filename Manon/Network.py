import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Complete.Transformer.LearnableModule import LearnableParameters


class LuongAttention(nn.Module):

    def __init__(self, hidden_dim, scaled=True, dropout=0.0):
        super().__init__()
        self.scaled = scaled
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h, T_hist=5):
        B, T, H = h.shape

        # Keys/Values
        K = self.W(h)
        V = h  # [B, T, H]
        Q = h  # [B, T, H]

        # Scores: [B, T, T] = Q @ K^T
        scores = torch.bmm(Q, K.transpose(1, 2))
        if self.scaled:
            scores = scores / math.sqrt(H)

        # 1. Masque Causal standard (Garde le triangle inférieur)
        mask_future = torch.tril(torch.ones(T, T, device=h.device, dtype=torch.bool))
        # 2. Masque de Fenêtre (Garde ce qui est au-dessus de la limite -T_hist)
        # diagonal=-K signifie qu'on part de la K-ième diagonale en dessous de la centrale
        mask_past = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=-(T_hist-1))
        # 3. Intersection : Il faut satisfaire les DEUX conditions
        valid_mask = mask_future & mask_past
        # 4. Application
        scores = scores.masked_fill(~valid_mask, float("-inf"))

        # Softmax attention
        attn = F.softmax(scores, dim=-1)  # [B, T, T]
        attn = attn.nan_to_num(0.0)  # guard against all -inf rows
        attn = self.dropout(attn)

        # Context
        context = torch.bmm(attn, V)  # [B, T, H]
        return context


class MemoryUpdateLSTMWithAttention(nn.Module):
    def __init__(
            self,
            input_dim_1,
            input_dim_2,
            hidden_dim,
            output_dim,
            num_layers=1,
            dropout=0.0,
            mem_length=5,
            attn_dropout=0.0,
            use_layernorm=True,
            use_mlp_head=True,
            pack_with_mask=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mem_length = mem_length
        self.pack_with_mask = pack_with_mask
        self.use_layernorm = use_layernorm
        self.use_mlp_head = use_mlp_head

        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=False, )

        self.embedding_1 = nn.Sequential(
            nn.Linear(input_dim_1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embedding_2 = nn.Sequential(
            nn.Linear(input_dim_2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.merge_embeddings = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(2*hidden_dim, hidden_dim),
        )
        self.combiner = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.next_token = LearnableParameters(torch.normal(0, 1, [1, 1, hidden_dim]))

        if use_layernorm:
            self.post_rnn_ln = nn.LayerNorm(hidden_dim)

        self.attn = LuongAttention(hidden_dim=hidden_dim, scaled=True, dropout=attn_dropout, )

        if use_mlp_head:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.head = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        # LSTM orthogonal init
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Linear layers
        def init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_linear)


    def forward(self, input_1, input_2, next_mask=None):
        if next_mask is None:
            next_mask = torch.zeros(*input_1.shape[:-1], 1)
        x_1 = self.embedding_1(input_1)
        x_2 = self.embedding_2(input_2)
        x_2 = x_2 * (1 - next_mask) + next_mask * self.next_token()

        x = self.merge_embeddings(torch.cat([x_1, x_2],dim=-1))  # [B, T, input_dim]

        rnn_out, _ = self.lstm(x)  # [B, T, H]

        if self.use_layernorm:
            rnn_out = self.post_rnn_ln(rnn_out)

        context = self.attn(rnn_out, T_hist=self.mem_length)  # [B,T,H], [B,T,T]
        combined = self.combiner(torch.cat([rnn_out, context], dim=-1))  # [B, T, 2H]
        next_dist = torch.norm(combined - self.next_token(), dim=-1) / math.sqrt(self.hidden_dim)
        pred = self.head(combined)  # [B, T, output_dim]

        return pred, next_dist

    # lightweight single-step Luong attention for one query q_t against past H (decoder states)
    def _attend_step(self, H_all, q_t):
        # H_all: [B, L, H], q_t: [B, H]
        K = self.attn.W(H_all)  # [B, L, H]
        V = H_all
        # scores: [B, 1, L] = q_t @ K^T
        scores = torch.bmm(q_t.unsqueeze(1), K.transpose(1, 2))
        if self.attn.scaled:
            scores = scores / math.sqrt(K.size(-1))
        # causal already ensured by using only past states in H_all
        attn = torch.softmax(scores, dim=-1)  # [B,1,L]
        attn = self.attn.dropout(attn)
        ctx = torch.bmm(attn, V).squeeze(1)  # [B,H]
        return ctx

    def step(self, input_1, input_2, hidden=None, H_past=None, next_mask=None):
        """
        Returns:
          y_t:    [B, output_dim]
          hidden: (h,c) new LSTM state
          H_new:  [B, L+1, H] updated decoder hidden history (pre-LN)
        """
        # run one LSTM step
        if next_mask is None:
            next_mask = torch.zeros(*input_1.shape[:-1], 1, 1)
        x_1 = self.embedding_1(input_1.unsqueeze(1))
        x_2 = self.embedding_2(input_2.unsqueeze(1))
        x_2 = x_2 * (1 - next_mask) + next_mask * self.next_token()

        x = self.merge_embeddings(torch.cat([x_1, x_2], dim=-1)) # [B,1,F]

        rnn_out, hidden = self.lstm(x, hidden)  # rnn_out: [B,1,H]
        h_t = rnn_out.squeeze(1)  # [B,H]

        # layernorm (keep cache in the same space the attention sees during forward)
        if self.use_layernorm:
            h_t = self.post_rnn_ln(h_t)  # [B,H]

        # update decoder-state cache
        if H_past is None:
            H_new = h_t.unsqueeze(1)  # [B,1,H]
        else:
            H_new = torch.cat([H_past[:, -(self.mem_length - 1):], h_t.unsqueeze(1)], dim=1)  # [B,L+1,H]

        # Luong attention over [past + current] decoder states (causal by construction)
        ctx_t = self._attend_step(H_new, h_t)  # [B,H]

        # head
        combined_t = self.combiner(torch.cat([h_t, ctx_t], dim=-1).unsqueeze(1))  # [B,1,2H]
        next_dist = torch.norm(combined_t - self.next_token(), dim=-1) / math.sqrt(self.hidden_dim)
        y_t = self.head(combined_t).squeeze(1)  # [B,output_dim]
        return y_t, hidden, H_new, next_dist

if __name__ == '__main__':
    N = MemoryUpdateLSTMWithAttention(10, 10, 64, 5)
    x1 = torch.normal(0, 1, (40, 20, 10))
    x2 = torch.normal(0, 1, (40, 20, 10))
    y1, next_dist_1 = N(x1, x2)

    y_list = []
    next_dist_list = []
    hidden = None
    H_past = None
    for k in range(x1.shape[1]):
        x1_k = x1[:, k]
        x2_k = x2[:, k]
        y_k, hidden, H_past, next_dist = N.step(x1_k, x2_k, hidden, H_past)
        y_list.append(y_k.unsqueeze(1))
        next_dist_list.append(next_dist)
    y2 = torch.cat(y_list, dim=1)
    next_dist_2 = torch.cat(next_dist_list, dim=1)

    print(y2[0, :, 0])
    print(y1[0, :, 0])

    print(next_dist_1[0])
    print(next_dist_2[0])