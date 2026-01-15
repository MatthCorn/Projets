import torch
from Inter.Model.DataMaker import GetData as GD

def GetData(d_in, n_pulse_plateau, n_sat, n_mes, len_in, len_out, n_data_training, n_data_validation=1000, sensitivity=0.1,
            weight_f=None, weight_l=None, bias='none', std_min=1., std_max=5., mean_min=-10., mean_max=10.,
            distrib='log', plot=False, save_path=None, parallel=False, max_inflight=None):

    [(TrainingInput, TrainingOutput, TrainingMasks, TrainingStd),
     (ValidationInput, ValidationOutput, ValidationMasks, ValidationStd)] = GD(
        d_in=d_in,
        n_pulse_plateau=n_pulse_plateau,
        n_sat=n_sat,
        n_mes=n_mes,
        len_in=len_in,
        len_out=len_out,
        n_data_training=n_data_training,
        n_data_validation=n_data_validation,
        sensitivity=sensitivity,
        bias=bias,
        mean_min=mean_min,
        mean_max=mean_max,
        std_min=std_min,
        std_max=std_max,
        distrib=distrib,
        weight_f=weight_f,
        weight_l=weight_l,
        type='complete',
        save_path=save_path,
        parallel=parallel,
        max_inflight=max_inflight
    )


def build_decoder_input_output_parallel(TrainingInput, TrainingOutput, TrainingMasks):
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
    ################################################################################################################################################
    # DIMENSIONS :
    device = TrainingInput.device
    B, len_in, d_in = TrainingInput.shape  # [B, len_in, d_in]
    _, len_out, d_out = TrainingOutput.shape  # [B, len_out, d_out]

    # d_in : Input Dimention
    # d_out = d_in +1 : Output Dimention

    # len_in : Length of Input Sequence
    # len_out : Length of Output Sequence
    # w/ len_in < len_out = len_in * 2
    ################################################################################################################################################

    # STEP 1 : Compute TOA_input and TOE_output
    ################################################################################################################################################

    # TOA INPUT :
    TOA_input = torch.arange(len_in, device=device).float().expand(B, -1)  # [B, len_in]

    # TOE OUTPUT :
    offset = TrainingOutput[:, :, -1]  # [B, len_out]
    LI = TrainingOutput[:, :, -2]  # [B, len_out]
    TOA_output = torch.arange(len_out, device=device).float().expand(B, -1) - offset  # [B, len_out]
    CONSTANT = 2.0
    TOE_output = TOA_output + LI  # + CONSTANT # [B, len_out]

    # STEP 2: TOA + END_TOKEN
    ################################################################################################################################################
    valid_output_len = TrainingMasks[1].sum(dim=1).long().squeeze(
        1) - 1  # [B] => correspond to mult_mask sum for each sequence in batch
    batch_indices = torch.arange(B, device=device)
    last_TOE = TOE_output[batch_indices, valid_output_len - 1]  # [B] => last valid TOE
    END_TOKEN = last_TOE + 1.0  # [B]
    TOA_input_with_end = torch.cat([TOA_input, END_TOKEN.unsqueeze(1)], dim=1)  # [B, len_in + 1]

    # STEP 3: Mask padded TOE values
    ################################################################################################################################################
    TOE_PAD_VALUE = float(len_out + 1)  # Value to pad TOE output = one extra time step than len_out
    # (for now padded with 0s - will cause issue in sorting tho, need padding after sorting at the end)
    mask = torch.arange(len_out, device=device).unsqueeze(0) > (valid_output_len.unsqueeze(1) - 1)  # [B, len_out]
    TOE_output = TOE_output.masked_fill(mask, TOE_PAD_VALUE)  # [B, len_out]

    # Create validity mask: 1 for valid TOEs, 0 for padded values
    # This mask will be used to filter out the padded values in the TOE output sequence => To get only the valid TOE outputs
    TOE_valid_mask = (~mask).float()  # [B, len_out]

    # STEP 4: Time concatenation and sorting :
    ################################################################################################################################################
    combined_times = torch.cat([TOA_input_with_end, TOE_output], dim=1)  # [B, T] w/ T = (len_in + 1) + (len_out)
    sorted_indices = torch.argsort(combined_times, dim=1)  # [B, T]

    # STEP 5: Build A (input + end + output)
    ################################################################################################################################################
    # Define FLAG as index of end token
    FLAG = len_in
    end_token_vec = torch.full((B, 1, d_in), float(FLAG), device=device)
    if d_out > d_in:
        pad_dim = d_out - d_in
        padded_input = torch.cat([TrainingInput, torch.zeros(B, len_in, pad_dim, device=device)], dim=2)
        padded_end_token_vec = torch.cat([end_token_vec, torch.zeros(B, 1, pad_dim, device=device)], dim=2)
    else:
        padded_input = TrainingInput  # [B, len_in, d_out]
        padded_end_token_vec = end_token_vec  # [B, 1, d_out]

    arr_A = torch.cat([padded_input, padded_end_token_vec, TrainingOutput],
                      dim=1)  # [B, T, d_out] w/ T = len_in (input) + 1 (end token vec) + len_out (output)

    # STEP 6: Compute MULT + ADD MASK
    ################################################################################################################################################
    # MULT MASK : Marks valid positions in the sequence (No padding)
    valid_prefix = torch.ones(B, len_in + 1, device=device)
    mult_mask = torch.cat([valid_prefix, TOE_valid_mask], dim=1)  # [B, T]

    # ADD MASK : Marks positions where the next step of the end of the sequence is not a valid output
    shifted = torch.roll(mult_mask, shifts=1, dims=1)
    shifted[:, 0] = mult_mask[:, 0]
    add_mask = ((shifted == 1) & (mult_mask == 0)).cumsum(dim=1).eq(1).float()  # [B, T]

    # STEP 7: Compute array B (previous pulse index)
    ################################################################################################################################################
    # Add a padding token at the beginning of sorted_indices (-1 for the first position) and remove the last element
    pad = torch.full((B, 1), -1, dtype=torch.long, device=device)
    arr_B_raw = torch.cat([pad, sorted_indices[:, :-1]], dim=1)
    arr_B = arr_B_raw.clone()
    end_token_mask = add_mask.bool()  # Is this necessary?
    arr_B[end_token_mask] = sorted_indices[end_token_mask]

    # STEP 8: Compute array C (current input at each step)
    ################################################################################################################################################
    input_mask = (arr_B < FLAG)
    final_indices = torch.where(input_mask, arr_B + 1, arr_B)  # [B, T]
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, final_indices.shape[1])
    arr_C = arr_A[batch_idx, final_indices]  # [B, T, d_out]

    # STEP 9: Compute array D (0=input, 1=output)
    ################################################################################################################################################
    arr_D = (arr_B >= FLAG).long()  # [B, T]

    # STEP 10: Compute array E (index in input sequence for current step)
    ################################################################################################################################################
    cumsum_input = torch.cumsum((arr_D == 0).long(), dim=1)
    arr_E = torch.clamp(cumsum_input - 1, min=0)  # [B, T]

    # STEP 11: Build decoder input + output tensors
    ################################################################################################################################################
    current_inputs = arr_A[batch_idx, arr_E]  # [B, T, d_out]
    A_padded = torch.cat([torch.zeros(B, 1, d_out, device=device), arr_A], dim=1)
    prev_preds = A_padded[batch_idx, arr_B + 1]  # [B, T, d_out]
    decoder_output_tensor = prev_preds * arr_D.unsqueeze(-1).float()
    decoder_input_tensor = torch.stack([current_inputs, decoder_output_tensor], dim=2)  # [B, T, 2, d_out]

    # STEP 11.5: Δt_sinceEnd with strict masks; Δt = TOA_current - TOE_prev (no normalization)

    ################################################################################################################################################
    FLAG = len_in  # end-token position in concatenated timeline
    T_total = sorted_indices.shape[1]
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, T_total)

    # 1) Current element classification (STRICT: outputs are strictly > FLAG)
    curr_idx = sorted_indices  # [B, T]
    curr_is_input = (curr_idx < FLAG)  # [B, T]
    curr_is_end = (curr_idx == FLAG)  # [B, T]
    curr_is_output = (curr_idx > FLAG)  # [B, T] # <-- strictly greater

    # 2) Previous element classification (STRICT: previous real output strictly > FLAG)
    prev_idx = arr_B  # [B, T]
    prev_is_output = (prev_idx > FLAG)  # [B, T]
    prev_out_idx = (prev_idx - (FLAG + 1)).clamp(min=0, max=len_out - 1)  # [B, T]

    # safer gather: zero everywhere, then fill only where prev_is_output
    TOE_prev = torch.zeros(B, T_total, device=device, dtype=TOE_output.dtype)
    TOE_prev[prev_is_output] = TOE_output[batch_idx[prev_is_output], prev_out_idx[prev_is_output]]  # [B, T]

    # 3) Build TOA_current by type:
    # inputs -> TOA_input[arr_E]
    # end -> END_TOKEN
    # outputs -> END_TOKEN (we'll still subtract TOE_prev per your rule)
    arr_E_safe = arr_E.clamp(min=0, max=len_in - 1)
    TOA_inputs_only = TOA_input[batch_idx, arr_E_safe]  # [B, T]
    TOA_current = torch.zeros_like(TOE_prev)
    TOA_current = torch.where(curr_is_input, TOA_inputs_only, TOA_current)
    TOA_current = torch.where(curr_is_end, END_TOKEN.unsqueeze(1), TOA_current)
    TOA_current = torch.where(curr_is_output, END_TOKEN.unsqueeze(1), TOA_current)

    # 4) Δt = TOA_current - TOE_prev for ALL steps (inputs, end, outputs)
    # (Before any output exists: TOE_prev=0 -> first inputs give 0,1,2,3,4,...; at END: END-0=10)
    delta_scalar = TOA_current - TOE_prev  # [B, T] <-- single scalar per step

    ##### NORMALIZATION STRATEGIES #####

    # --- Per-sequence normalization over valid steps only ---
    # valid positions (exclude padding); keep end+outputs included since we want their scale too
    valid_mask = mult_mask.float()  # [B, T], 1 for valid timeline positions

    eps = 1e-6
    denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
    mean = (delta_scalar * valid_mask).sum(dim=1, keepdim=True) / denom
    var = ((delta_scalar - mean) ** 2 * valid_mask).sum(dim=1, keepdim=True) / denom
    std = (var + eps).sqrt()

    # normalized Δt; zero-out padded positions
    delta_norm = ((delta_scalar - mean) / (std + eps)) * valid_mask  # [B, T]

    # --- Build return tensors with normalized Δt ---
    T_total = sorted_indices.shape[1]

    # A) 3-channel version (for compatibility): [x_i, p_{i-1}, Δt_norm]
    decoder_input_tensor = torch.stack([current_inputs, decoder_output_tensor], dim=2)  # [B, T, 2, d_out]
    delta_expanded = delta_norm.unsqueeze(-1).unsqueeze(-1).expand(B, T_total, 1, d_out)
    decoder_input_tensor = torch.cat([decoder_input_tensor, delta_expanded], dim=2)  # [B, T, 3, d_out]

    # B) Flattened version with true scalar Δt_norm at the end
    decoder_input_with_dt = torch.cat([current_inputs, decoder_output_tensor, delta_norm.unsqueeze(-1)],
                                      dim=-1)  # [B, T, 2*d_out + 1] """

    # === STEP 12: Apply mult_mask and shift output ===
    ################################################################################################################################################
    mult_mask_expanded = mult_mask.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
    decoder_input_tensor *= mult_mask_expanded  # Is this necessary?
    decoder_output_tensor *= mult_mask.unsqueeze(-1)  # [B, T, d_out]
    decoder_output_tensor = torch.cat([decoder_output_tensor[:, 1:], decoder_output_tensor[:, :1]], dim=1)

    mult_mask_sum = mult_mask.sum(dim=1)  # [B] — sum of valid positions for each batch element

    # REMOVE PADDING :
    ################################################################################################################################################
    """ 
    # Slice per-batch using list comprehension (efficient, avoids for-loop body)
    decoder_input_list = [decoder_input_tensor[b, :int(mult_mask_sum[b].item())] for b in range(B)]
    decoder_output_list = [decoder_output_tensor[b, :int(mult_mask_sum[b].item())] for b in range(B)]

    # Stack the list to create final tensors
    print("\n=== DECODER INPUT TENSOR ===")
    print(decoder_input_list[0])  # Print first batch for verification
    print("\n=== DECODER OUTPUT TENSOR ===")
    print(decoder_output_list[0])  # Print first batch for verification """

    # WITHOUT DELTA TIME :
    # return decoder_input_tensor, decoder_output_tensor, mult_mask, mult_mask_sum, arr_D

    # WITH DELTA TIME :
    return decoder_input_tensor, decoder_output_tensor, mult_mask, mult_mask_sum, arr_D, decoder_input_with_dt

