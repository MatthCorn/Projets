from typing import Tuple, Optional
import torch

@torch.no_grad()
def build_decoder_input_output_parallel(
    TrainingInput: torch.Tensor,    # [B, len_in, d_in]
    TrainingOutput: torch.Tensor,   # [B, len_out, d_out]
    TrainingMasks,                  # only TrainingMasks[1] (output mask) is used
    *,
    fill_prev_with_pad: bool = False,
    pad_vector: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    '''
    Important Info : 
    INPUT :
    - TrainingInput: [B, len_in, d_in]
    - TrainingOutput: [B, len_out, d_out]

    OUTPUT :
    # This is the input w/ [current (pulse) input, previous output prediction, delta_time]
    decoder_input_tensor:  [B, T, 3, d_out]  w/ T = (len_in + 1) + (len_out)
    
    # This is the output w/ [decoder output prediction]
    decoder_output_tensor: [B, T, d_out]
    
    # This is the mask for valid positions in the sequence (to remove the padding-> since we need the same length in tensor)
    mult_mask:             [B, T]
    mult_mask_sum:         [B]

    arr_D:                 [B, T] 
    Note : Array D - markes if the current step is an "output pulse prediction" (1) or a "next tokne" (0)
    '''
    device = TrainingInput.device
    B, len_in, d_in   = TrainingInput.shape
    _, len_out, d_out = TrainingOutput.shape

    out_mask = TrainingMasks[1]
    if out_mask.dim() == 3 and out_mask.size(-1) == 1:
        out_mask = out_mask.squeeze(-1)                      # [B, L?]
    out_mask = out_mask.to(device=device, dtype=torch.long)

    # If mask includes a START column (len_out+1), drop it:
    if out_mask.size(1) == len_out + 1:
        out_mask = out_mask[:, 1:]
    elif out_mask.size(1) != len_out:
        raise ValueError(
            f"Output mask length mismatch: got {out_mask.size(1)} but expected {len_out} or {len_out+1} (with START)."
        )

    TOA_input  = torch.arange(len_in,  device=device, dtype=torch.float32).expand(B, -1)  # [B, len_in]
    offset     = TrainingOutput[:, :, -1]  # [B, len_out]
    LI         = TrainingOutput[:, :, -2]  # [B, len_out]
    TOA_output = torch.arange(len_out, device=device, dtype=torch.float32).expand(B, -1) - offset
    TOE_output = TOA_output + LI  # [B, len_out]

    valid_output_len = out_mask.sum(dim=1).long()             # [B]
    batch_idx = torch.arange(B, device=device)
    safe_idx = (valid_output_len - 1).clamp(min=0)
    last_TOE = torch.where(valid_output_len > 0, TOE_output[batch_idx, safe_idx], torch.zeros_like(safe_idx, dtype=TOE_output.dtype))
    END_TOKEN_TIME = last_TOE + 1.0  # [B]
    TOA_input_with_end = torch.cat([TOA_input, END_TOKEN_TIME.unsqueeze(1)], dim=1)  # [B, len_in+1]

    # Mask padded TOE values and merge
    TOE_PAD_VALUE = float(len_out + 1)
    pad_mask = torch.arange(len_out, device=device).unsqueeze(0) >= valid_output_len.unsqueeze(1)  # [B, len_out]
    TOE_output_masked = TOE_output.masked_fill(pad_mask, TOE_PAD_VALUE)

    combined_times = torch.cat([TOA_input_with_end, TOE_output_masked], dim=1)  # [B, len_in+1+len_out]
    sorted_indices = torch.argsort(combined_times, dim=1)                       # [B, T]
    T = combined_times.size(1)

    FLAG = len_in  # END is at absolute position len_in in concatenation
    if d_out > d_in:
        pad_dim = d_out - d_in
        padded_input = torch.cat([TrainingInput,
                                  torch.zeros(B, len_in, pad_dim, device=device, dtype=TrainingInput.dtype)], dim=2)
    else:
        padded_input = TrainingInput[..., :d_out]
    end_token_vec = torch.zeros(B, 1, d_out, device=device, dtype=TrainingInput.dtype)
    end_token_vec[:, :, 0] = float(FLAG)  # optional channel marker
    A = torch.cat([padded_input, end_token_vec, TrainingOutput], dim=1)  # [B, len_in+1+len_out, d_out]


    # Unsorted validity: inputs+END are valid(1), outputs follow out_mask
    valid_prefix = torch.ones(B, len_in + 1, device=device, dtype=torch.long)
    mult_mask_unsorted = torch.cat([valid_prefix, out_mask], dim=1)  # [B, T] (unsorted order)
    # Sort it with the same permutation used for times:
    mult_mask_sorted = mult_mask_unsorted.gather(1, sorted_indices)  # [B, T]
    mult_mask_sum = mult_mask_sorted.sum(dim=1)                      # [B]

    curr_idx = sorted_indices                                         # [B, T] (absolute indices in A)
    pad_prev = torch.full((B, 1), -1, device=device, dtype=torch.long)
    arr_B = torch.cat([pad_prev, sorted_indices[:, :-1]], dim=1)     # previous absolute index in A  [B, T]
    arr_D = (curr_idx > FLAG).long()                                  # 1 iff CURRENT step is an OUTPUT [B, T]

    # Current input index (how many inputs seen so far)
    cumsum_input = torch.cumsum((curr_idx < FLAG).long(), dim=1)      # [B, T]
    arr_E = (cumsum_input - 1).clamp(min=0, max=len_in - 1)           # [B, T]
    batch_idx2 = torch.arange(B, device=device).unsqueeze(1).expand(B, T)
    curr_x = A[batch_idx2, arr_E]                                     # [B, T, d_out]

    
    # Previous output stream prev_y
    #################################################################################################################################################
    prev_is_output = (arr_B > FLAG)                                   # [B, T]
    prev_out_idx = (arr_B - (FLAG + 1)).clamp(min=0, max=len_out - 1) # [B, T] index w.r.t. outputs
    prev_y = torch.zeros(B, T, d_out, device=device, dtype=TrainingOutput.dtype)
    if prev_is_output.any():
        bsel = batch_idx2[prev_is_output]
        tsel = prev_out_idx[prev_is_output]
        prev_y[prev_is_output] = TrainingOutput[bsel, tsel, :]

    if fill_prev_with_pad:
        assert pad_vector is not None and pad_vector.numel() == d_out, \
            "pad_vector[d_out] required when fill_prev_with_pad=True"
        pad_row = pad_vector.to(device=prev_y.device, dtype=prev_y.dtype).view(1, 1, -1)
        prev_y = torch.where(prev_is_output.unsqueeze(-1), prev_y, pad_row.expand(B, T, d_out))

   
    #################################################################################################################################################
    TOE_prev = torch.zeros(B, T, device=device, dtype=TrainingOutput.dtype)
    if prev_is_output.any():
        TOE_prev[prev_is_output] = TOE_output[batch_idx2[prev_is_output], prev_out_idx[prev_is_output]]

    curr_is_input  = (curr_idx < FLAG)
    curr_is_end    = (curr_idx == FLAG)
    curr_is_output = (curr_idx > FLAG)

    TOA_inputs_only = TOA_input[batch_idx2, arr_E.clamp(min=0, max=len_in - 1)]
    TOA_current = torch.zeros_like(TOE_prev)
    TOA_current = torch.where(curr_is_input,  TOA_inputs_only, TOA_current)
    TOA_current = torch.where(curr_is_end,    END_TOKEN_TIME.unsqueeze(1), TOA_current)
    TOA_current = torch.where(curr_is_output, END_TOKEN_TIME.unsqueeze(1), TOA_current)

    delta_scalar = TOA_current - TOE_prev                              # [B, T]

    valid = mult_mask_sorted.float()                                    # [B, T]
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)               # [B, 1]
    mean  = (delta_scalar * valid).sum(dim=1, keepdim=True) / denom     # [B, 1]
    var   = ((delta_scalar - mean)**2 * valid).sum(dim=1, keepdim=True) / denom
    std   = (var + 1e-6).sqrt()
    delta_norm = ((delta_scalar - mean) / (std + 1e-6)) * valid         # [B, T]


    delta_broadcast = delta_norm.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1, d_out)
    decoder_input_tensor = torch.stack([curr_x, prev_y], dim=2)                 # [B, T, 2, d_out]
    decoder_input_tensor = torch.cat([decoder_input_tensor, delta_broadcast], 2)# [B, T, 3, d_out]

    decoder_output_tensor = prev_y.clone()                                      # [B, T, d_out]
    decoder_input_with_dt = torch.cat([curr_x, prev_y, delta_norm.unsqueeze(-1)], dim=-1)  # [B, T, 2*d_out+1]

    """ssert decoder_input_tensor.shape  == (B, T, 3, d_out)
    assert decoder_output_tensor.shape == (B, T, d_out)
    assert decoder_input_with_dt.shape == (B, T, 2*d_out + 1)
    assert mult_mask_sorted.shape      == (B, T)
    assert arr_D.shape                 == (B, T) """

    return (
        decoder_input_tensor,   # [B, T, 3, d_out]
        decoder_output_tensor,  # [B, T, d_out]
        mult_mask_sorted,       # [B, T]  (sorted order)
        mult_mask_sum,          # [B]
        arr_D,                  # [B, T]
        decoder_input_with_dt,  # [B, T, 2*d_out+1]
    )
