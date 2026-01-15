import torch
import torch.nn as nn

# AUTOREGRESSIVE INFERENCE :
# ################################################################################################################################################ 
@torch.no_grad()
def ar_decode_lstm_luong_cached(
    model: nn.Module,
    decoder_input_with_dt: torch.Tensor,   # [B, T, 2*d_out+1] = [curr, prev(ignored at test), dt]
    key_padding_mask: torch.Tensor,        # [B, T] (1=valid, 0=pad)  — used for early stop length only
    d_out: int,
    start_token: torch.Tensor,             # [B, d_out]
    arr_D: torch.Tensor = None,            # [B, T] gate: 1 if previous output is meaningful
    end_token: torch.Tensor = None,
    end_eps: float = 0.05,
):
    device = decoder_input_with_dt.device
    B, T, F = decoder_input_with_dt.shape
    curr = decoder_input_with_dt[..., :d_out]   # [B,T,d_out]
    dt   = decoder_input_with_dt[..., -1:]      # [B,T,1]

    preds = torch.zeros(B, T, d_out, device=device)
    prev_out = start_token.to(device)

    # LSTM + attention caches
    hidden = None         # (h,c)
    H_past = None         # [B,L,H]

    done_mask = torch.zeros(B, dtype=torch.bool, device=device)

    for t in range(T):
        # gate previous output if provided
        if arr_D is not None:
            arrDt = arr_D[:, t].unsqueeze(-1).to(prev_out.dtype)   # [B,1]
            prev_feat = prev_out * arrDt
        else:
            prev_feat = prev_out

        step_in = torch.cat([curr[:, t, :], prev_feat, dt[:, t, :]], dim=-1)  # [B, 2*d_out+1]

        if done_mask.any():
            step_in = step_in.clone()
            step_in[done_mask] = 0.0

        y_t, hidden, H_past = model.step(step_in, hidden, H_past)   # fast O(1) LSTM; O(t) attention

        # write prediction but keep finished samples frozen at end_token (if any)
        if done_mask.any() and end_token is not None:
            y_t = y_t.clone()
            y_t[done_mask] = end_token.to(device)

        preds[:, t, :] = y_t
        prev_out = y_t

        # early-stop per sample (optional, last dim by convention)
        if end_token is not None:
            end_dim = -1
            is_end = (y_t[:, end_dim] - end_token[end_dim].to(device)).abs() < end_eps
            done_mask = done_mask | is_end
            if done_mask.all():
                break

    return preds

# SCHEDULE SAMPLING (gradient-safe)  : 
def scheduled_sampling_decode_lstm_luong_cached(
    model: nn.Module,
    decoder_input_with_dt: torch.Tensor,   # [B, T, 2*d_out+1] = [curr, prev_gt(masked), dt]  (same builder OK)
    key_padding_mask: torch.Tensor,        # [B, T]
    d_out: int,
    arr_D: torch.Tensor,                   # [B, T]
    p_tf: float,                           # prob to use GT prev
    start_token: torch.Tensor,             # [B, d_out]
):
    device = decoder_input_with_dt.device
    B, T, F = decoder_input_with_dt.shape

    curr = decoder_input_with_dt[..., :d_out]               # [B,T,d_out]
    prev_gt_stream = decoder_input_with_dt[..., d_out:2*d_out]  # [B,T,d_out] (already gated upstream)
    dt   = decoder_input_with_dt[..., -1:]                  # [B,T,1]

    preds = []
    prev_out = start_token                                  # [B,d_out]
    hidden = None
    H_past = None

    for t in range(T):
        arrDt = arr_D[:, t].unsqueeze(-1).to(prev_out.dtype)       # [B,1]

        use_tf = (torch.rand(B, 1, device=device) < p_tf).to(prev_out.dtype)  # [B,1]
        prev_model_masked = (prev_out.detach()) * arrDt            # detach model prev for stability
        prev_chosen = use_tf * prev_gt_stream[:, t, :] + (1 - use_tf) * prev_model_masked  # [B,d_out]

        step_in = torch.cat([curr[:, t, :], prev_chosen, dt[:, t, :]], dim=-1)  # [B,2*d_out+1]
        y_t, hidden, H_past = model.step(step_in, hidden, H_past)               # keep graph through step

        preds.append(y_t.unsqueeze(1))
        prev_out = y_t

    return torch.cat(preds, dim=1)  # [B,T,d_out]
################################################################################################################################################

