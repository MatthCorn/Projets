import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LuongAttention(nn.Module):
    """
    Luong-style attention over a sequence of decoder hidden states.

    score(q_t, k_s) = q_t^T W k_s   (mode='general') or q_t^T k_s     (mode='dot')

    Args:
        hidden_dim:  feature size H of the RNN hidden states
        mode:        'general' or 'dot'
        causal:      if True, each position t attends only to s <= t
        scaled:      divide scores by sqrt(H) for stability
        dropout:     dropout applied on attention probabilities
    """
    def __init__(self, hidden_dim: int, mode: str = "general", causal: bool = True, scaled: bool = True, dropout: float = 0.0,):
        super().__init__()
        assert mode in {"general", "dot"}
        self.mode = mode
        self.causal = causal
        self.scaled = scaled
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if mode == "general":
            self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.register_buffer("_dummy", torch.tensor(0), persistent=False)  # device helper

    def forward( self, h: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H = h.shape

        # Keys/Values
        K = self.W(h) if self.mode == "general" else h  # [B, T, H]
        V = h                                           # [B, T, H]
        Q = h                                           # [B, T, H]

        # Scores: [B, T, T] = Q @ K^T
        scores = torch.bmm(Q, K.transpose(1, 2))
        if self.scaled:
            scores = scores / math.sqrt(H)

        # Causal mask (block future positions)
        if self.causal:
            causal = torch.tril(
                torch.ones(T, T, device=h.device, dtype=torch.bool)
            )
            scores = scores.masked_fill(~causal, float("-inf"))

        # Key padding mask: block attending to padded positions
        if key_padding_mask is not None:
            # valid keys: 1, pads: 0
            valid = key_padding_mask.to(torch.bool).unsqueeze(1)  # [B,1,T]
            scores = scores.masked_fill(~valid, float("-inf"))

        # Softmax attention
        attn = F.softmax(scores, dim=-1)              # [B, T, T]
        attn = attn.nan_to_num(0.0)                   # guard against all -inf rows
        attn = self.dropout(attn)

        # Context
        context = torch.bmm(attn, V)                  # [B, T, H]
        return context, attn


class MemoryUpdateLSTMWithAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        attn_mode: str = "general",
        causal_attn: bool = True,
        attn_dropout: float = 0.0,
        use_layernorm: bool = True,
        use_mlp_head: bool = True,
        pack_with_mask: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pack_with_mask = pack_with_mask
        self.use_layernorm = use_layernorm
        self.use_mlp_head = use_mlp_head

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,dropout=dropout if num_layers > 1 else 0.0,bidirectional=False,)

        if use_layernorm:
            self.post_rnn_ln = nn.LayerNorm(hidden_dim)

        self.attn = LuongAttention(hidden_dim=hidden_dim,mode=attn_mode,causal=causal_attn,scaled=True,dropout=attn_dropout,)

        if use_mlp_head:
            self.head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.head = nn.Linear(2 * hidden_dim, output_dim)

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
        
    @staticmethod
    def _flatten_decoder_input(decoder_input: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
        - [B, T, input_dim] -passthrough (already flattened)
        - [B, T, 2, d_out]  -flattens to [B, T, 2*d_out]
        - [B, T, 3, d_out] (with Δt)-reduces Δt to scalar then [B, T, 2*d_out+1]
        """
        if decoder_input.dim() == 3:
            return decoder_input
        elif decoder_input.dim() == 4:
            B, T, C, D = decoder_input.shape
            if C == 2:
                return decoder_input.reshape(B, T, 2 * D)
            elif C == 3:
                x_cur  = decoder_input[:, :, 0, :]
                x_prev = decoder_input[:, :, 1, :]
                dt_map = decoder_input[:, :, 2, :]
                dt_scalar = dt_map.mean(dim=-1, keepdim=True)  # or dt_map[..., :1]
                return torch.cat([x_cur, x_prev, dt_scalar], dim=-1)  # [B,T,2*D+1]
            else:
                raise ValueError(f"Unsupported channel count C={C}")
        else:
            raise ValueError(f"Bad shape {decoder_input.shape}")


    def forward(self, decoder_input: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, return_attn: bool = False,):
        """
        Returns:
            pred:         [B, T, output_dim]
            attn_weights: [B, T, T] (if return_attn=True)
        """
        x = self._flatten_decoder_input(decoder_input)  # [B, T, input_dim]

        if self.pack_with_mask and key_padding_mask is not None:
            lengths = key_padding_mask.sum(dim=1).to(torch.int64).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths=lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=x.size(1)
            )  # [B, T, H]
        else:
            rnn_out, _ = self.lstm(x)  # [B, T, H]

        if self.use_layernorm:
            rnn_out = self.post_rnn_ln(rnn_out)

        context, attn_weights = self.attn(rnn_out, key_padding_mask)  # [B,T,H], [B,T,T]
        combined = torch.cat([rnn_out, context], dim=-1)              # [B, T, 2H]
        pred = self.head(combined)                                     # [B, T, output_dim]

        if return_attn:
            return pred, attn_weights
        return pred
    
     # lightweight single-step Luong attention for one query q_t against past H (decoder states)
    def _attend_step(self, H_all: torch.Tensor, q_t: torch.Tensor) -> torch.Tensor:
        # H_all: [B, L, H], q_t: [B, H]
        if hasattr(self.attn, "W") and self.attn.mode == "general":
            K = self.attn.W(H_all)           # [B, L, H]
        else:
            K = H_all
        V = H_all
        # scores: [B, 1, L] = q_t @ K^T
        scores = torch.bmm(q_t.unsqueeze(1), K.transpose(1, 2))
        if self.attn.scaled:
            scores = scores / math.sqrt(K.size(-1))
        # causal already ensured by using only past states in H_all
        attn = torch.softmax(scores, dim=-1)   # [B,1,L]
        attn = self.attn.dropout(attn)
        ctx = torch.bmm(attn, V).squeeze(1)    # [B,H]
        return ctx

    def step( self, x_t: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,H_past: Optional[torch.Tensor] = None,):
        """
        Returns:
          y_t:    [B, output_dim]
          hidden: (h,c) new LSTM state
          H_new:  [B, L+1, H] updated decoder hidden history (pre-LN)
        """
        # run one LSTM step
        x_t = x_t.unsqueeze(1)                # [B,1,F]
        rnn_out, hidden = self.lstm(x_t, hidden)   # rnn_out: [B,1,H]
        h_t = rnn_out.squeeze(1)                   # [B,H]

        # layernorm (keep cache in the same space the attention sees during forward)
        if self.use_layernorm:
            h_t = self.post_rnn_ln(h_t)            # [B,H]

        # update decoder-state cache
        if H_past is None:
            H_new = h_t.unsqueeze(1)                 # [B,1,H]
        else:
            H_new = torch.cat([H_past, h_t.unsqueeze(1)], dim=1)  # [B,L+1,H]


        # Luong attention over [past + current] decoder states (causal by construction)
        ctx_t = self._attend_step(H_new, h_t)      # [B,H]

        # head
        combined_t = torch.cat([h_t, ctx_t], dim=-1).unsqueeze(1)  # [B,1,2H]
        y_t = self.head(combined_t).squeeze(1)                     # [B,output_dim]
        return y_t, hidden, H_new
