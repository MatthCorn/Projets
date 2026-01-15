from math import sqrt
import torch

# LOSS FUNCTION :
################################################################################################################################################
# LOSS 1: SIMPLE MSE LOSS (< 1 end token verification)
def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()

# LOSS 2 : MASKED MSE LOSS
def masked_mse_loss(pred, target, target_mask, eps=1e-8):
    add_mask, mult_mask = target_mask 
    err = (pred - target) ** 2
    masked_err = err * mult_mask
    return masked_err.sum() / (mult_mask.sum() * pred.size(-1) + eps)

# LOSS 3.1 : NORMALIZED + MASKED MSE LOSS (per-feature variance normalization)
def normalized_mse_loss(pred, target, target_mask, eps=1e-8):
    add_mask, mult_mask = target_mask

    err = (pred - target) ** 2

    masked_target = target * mult_mask
    mean_per_feature = masked_target.sum(dim=(0, 1)) / (mult_mask.sum(dim=(0, 1)) + eps)
    centered_target = masked_target - mean_per_feature.view(1, 1, -1)
    var_per_feature = (centered_target ** 2).sum(dim=(0, 1)) / (mult_mask.sum(dim=(0, 1)) + eps)

    norm_err = err / (var_per_feature.view(1, 1, -1) + eps)

    masked_err = norm_err * mult_mask
    return masked_err.sum() / (mult_mask.sum() * pred.size(-1) + eps)

# LOSS 4.2 : # WEIGHTED + NORMALIZED + MASKED MSE LOSS (per-feature variance normalization with weights)
def weighted_normalized_mse_loss_2(pred, target, target_mask, loss_weights, eps=1e-8):
    add_mask, mult_mask = target_mask 
    err = (pred - target) ** 2 # MSE :

    # Feature variance for normalization :
    masked_target = target * mult_mask
    feature_var = (masked_target ** 2).sum(dim=(0, 1)) / (mult_mask.sum(dim=(0, 1)) + eps)
    feature_var = feature_var.view(1, 1, -1)

    # Normalized Loss : 
    norm_err = err / (feature_var + eps)

    # Weighted + Masked Loss :
    weighted_err = norm_err * loss_weights
    masked_err = weighted_err * mult_mask

    #return masked_err.sum() / (mult_mask.sum() + eps) / (loss_weights.sum() + eps)
    return masked_err.sum() / (mult_mask.sum() + eps) 

# LOSS 5 : RNN NORMALIZED LOSS (standardized error)
def rnn_std_normalized_mse(pred, target, target_mask, std, eps=1e-8):
    _, mult_mask = target_mask  # (B, L, 1)

    # Apply masking
    masked_pred = pred * mult_mask
    masked_target = target * mult_mask

    # Standardized squared error : std **2 = variance 
    normed_err = ((masked_pred - masked_target) / (std + eps)) ** 2

    return normed_err.sum() / (mult_mask.sum() * pred.size(-1) + eps)

# LOSS 6 : RNN WEIGHTED + NORMALIZED LOSS (STD - standardized error)
def rnn_weighted_std_normalized_mse(pred, target, target_mask, std, loss_weights=None, eps=1e-8):
    _, mult_mask = target_mask

    # Apply masking
    masked_pred = pred * mult_mask
    masked_target = target * mult_mask

    # Standardized squared error
    normed_err = ((masked_pred - masked_target) / (std + eps)) ** 2
    normed_err = normed_err * loss_weights  # (1, 1, D) broadcast

    return normed_err.sum() / (mult_mask.sum() * pred.size(-1) + eps)

# LOSS 7 : WEIGHTED + NORMALIZED + MASKED MSE LOSS with EOS penalty
def weighted_normalized_mse_with_eos_penalty(pred, target, target_mask, loss_weights, end_token_vec, alpha=1.0, eps=1e-8):
    add_mask, mult_mask = target_mask

    err = (pred - target) ** 2
    masked_target = target * mult_mask
    feature_var = (masked_target ** 2).sum(dim=(0, 1)) / (mult_mask.sum(dim=(0, 1)) + eps)
    feature_var = feature_var.view(1, 1, -1)
    norm_err = err / (feature_var + eps)
    weighted_err = norm_err * loss_weights.view(1, 1, -1)
    masked_err = weighted_err * mult_mask
    main_loss = masked_err.sum() / (mult_mask.sum() + eps) / (loss_weights.sum() + eps)

    # EOS loss using add_mask :
    eos_err = ((pred - end_token_vec) ** 2) / (feature_var + eps)
    eos_weighted = eos_err * loss_weights.view(1, 1, -1)
    eos_masked = eos_weighted * add_mask

    eos_loss = eos_masked.sum() / (add_mask.sum() + eps) / (loss_weights.sum() + eps)

    return main_loss + alpha * eos_loss

# LOSS 7.2 : WEIGHTED + NORMALIZED + MASKED MSE LOSS with EOS penalty
def weighted_normalized_mse_with_eos_penalty_2(pred, target, target_mask, loss_weights, end_token_vec, alpha=1.0, eps=1e-8):
    add_mask, mult_mask = target_mask

    err = (pred - target) ** 2
    masked_target = target * mult_mask
    feature_var = (masked_target ** 2).sum(dim=(0, 1)) / (mult_mask.sum(dim=(0, 1)) + eps)
    feature_var = feature_var.view(1, 1, -1)
    norm_err = err / (feature_var + eps)
    weighted_err = norm_err * loss_weights.view(1, 1, -1)
    masked_err = weighted_err * mult_mask
    #main_loss = masked_err.sum() / (mult_mask.sum() + eps) / (loss_weights.sum() + eps)
    main_loss = masked_err.sum() / (mult_mask.sum() + eps) 

    # EOS loss using add_mask :
    eos_err = ((pred - end_token_vec) ** 2) / (feature_var + eps)
    eos_weighted = eos_err * loss_weights.view(1, 1, -1)
    eos_masked = eos_weighted * add_mask

    #eos_loss = eos_masked.sum() / (add_mask.sum() + eps) / (loss_weights.sum() + eps)
    eos_loss = eos_masked.sum() / (add_mask.sum() + eps) 

    return main_loss + alpha * eos_loss