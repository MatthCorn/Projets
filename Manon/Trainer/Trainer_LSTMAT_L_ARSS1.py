################################################################################################################################################
import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
import os

from Inter.Model.DataMaker import GetData

from FWForecasting_OG.Library.LRScheduler import CosineScheduler
from FWForecasting_OG.Library.Parameters_SIMPLE import get_parameter
from FWForecasting_OG.Library.LOSS import weighted_normalized_mse_loss_1 as loss_fct
from FWForecasting_OG.Library.Visualization_LengthAdapter import plot_visuals
from FWForecasting_OG.Library.Save_Plot_File import save_file, plot_graph, plot_graph_with_min

from FWForecasting_OG.Library.Decoder_Input_Output import build_decoder_input_output_parallel
#from FWForecasting_OG.Library.Decoder_Input_Output1 import build_decoder_input_output_parallel

from FWForecasting_OG.Library.Helper_Autoregression import scheduled_sampling_decode_lstm_luong_cached, ar_decode_lstm_luong_cached

from FWForecasting_OG.Models.Models_LSTMAT_L_ARSS1 import MemoryUpdateLSTMWithAttention
################################################################################################################################################


# SETUP : 
################################################################################################################################################
###################################
Current_Folder = 'MODEL_FINAL'
Current_Model = 'LSTMAT_Luong_AUTORSS1'
Current_File = 'LSTMAT_L_ARSS1_TEST5_NTF_500K_imp_70s_15e'
###################################

print("CURRENT MODEL :", Current_Model)
print("CURRENT FILE : ", Current_File)

# VSC : 
""" local = os.path.join(os.path.abspath(__file__)[:os.path.abspath(__file__).index("Manon")], "Manon")
save_dir = os.path.join(local, Current_Folder, 'Save')

visuals_path = f'/Users/thales/Desktop/MANON/Manon2/{Current_Folder}/Graph/Visuals'
graph_path = f'/Users/thales/Desktop/MANON/Manon2/{Current_Folder}/Graph/LossCurves'   """

# RUCHE :
local = '/gpfs/workdir/lagardema/MANON/FWForecasting_OG' 
save_dir = os.path.join(local, Current_Folder, 'Save')

visuals_path = f'/gpfs/workdir/lagardema/MANON/FWForecasting_OG/{Current_Folder}/Graph/Visuals'
graph_path = f'/gpfs/workdir/lagardema/MANON/FWForecasting_OG/{Current_Folder}/Graph/LossCurves'


# VISUALIZATION - FILE NAME :
visuals_plot_file = f"{Current_File}_Visuals"
#loss_graph_file = f"{Current_File}6.1_loss_curve_MSELoss_Trainer_Normalized.png"
sqrt_loss_graph_file = f"{Current_File}_MSELoss_curve_SQRT.png"

loss_name_file = f"{Current_File}_END_TOKEN_MSELoss.png"
################################################################################################################################################

# SCHEDULE SAMPLING : 
################################################################################################################################################
# TEACHER FORCING RATIO :
def tf_ratio(step, total_steps, start=1.0, end=0.0, mode="linear"):
    if mode == "linear":
        return max(end, start - (start - end) * (step / max(1, total_steps)))
    elif mode == "cos":
        import math
        return end + (start - end) * 0.5 * (1 + math.cos(math.pi * step / max(1, total_steps)))
    elif mode == "exp":
        import math
        k = 5.0
        return end + (start - end) * math.exp(-k * step / max(1, total_steps))
    return start
################################################################################################################################################

# PARAMS :
################################################################################################################################################
param = {
    "len_in": 70,
    "len_out":90,
    "d_in": 10,
    "n_pulse_plateau": 5,
    "sensitivity": 0.1,
    "lr":  0.0036457468231616305,
    #"lr_option": {"value": 0.0001, "reset": "y", "type": "cos"},
    "mult_grad": 1000,
    "weight_decay": 1.0533653380275711e-06,
    "NDataT": 500000,
    "NDataV": 1000,
    "batch_size": 50,
    "n_epochs": 15,
    "distrib": "log",
    "error_weighting": "y",
    "max_lr": 5,
    "warmup": 5,
    "training_strategy": [{"mean": [-5, 5], "std": [0.2, 1]}]
}
################################################################################################################################################

# SHAPES :
################################################################################################################################################
d_in = param['d_in']
d_out = d_in + 1
batch_size = param['batch_size']
n_epochs = param['n_epochs']

weight_f = torch.tensor([1., 0.] + [0.] * (d_in - 3)).numpy()
weight_l = torch.tensor([0., 1.] + [0.] * (d_in - 3)).numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################################################################################################################################################


# DATA :
################################################################################################################################################
strategy = param["training_strategy"][0]
[(TrainingInput, TrainingOutput, TrainingMasks, _),(ValidationInput, ValidationOutput, ValidationMasks, _)] = GetData(
    d_in=d_in,
    n_pulse_plateau=param['n_pulse_plateau'],
    n_sat=param['n_pulse_plateau'],
    n_mes=160, 
    len_in=param['len_in'],
    len_out=param['len_out'],
    n_data_training=param['NDataT'],
    n_data_validation=param['NDataV'],
    sensitivity=param['sensitivity'],
    weight_f=weight_f,
    weight_l=weight_l,
    bias='none',
    mean_min=strategy['mean'][0],
    mean_max=strategy['mean'][1],
    std_min=strategy['std'][0],
    std_max=strategy['std'][1],
    type='complete',
    distrib=param['distrib'],
    save_path='Manon2/Data',
    parallel=True
)
################################################################################################################################################


# BUILD DECODER SEQUENCES 
################################################################################################################################################
'''
    Important Info : 
    INPUT :
    - TrainingInput: [B, len_in, d_in]
    - TrainingOutput: [B, len_out, d_out]

    OUTPUT :
    decoder_input_tensor:  [B, T, 2, d_out]  w/ T = (len_in + 1) + (len_out)
    decoder_output_tensor: [B, T, d_out]

    mult_mask:             [B, T]
    mult_mask_sum:         [B]

    arr_D:                 [B, T] 
    Note : Array D marks if the current step is an "output pulse prediction" (1) or a "next tokne" (0)
'''
decoder_input_tensor, decoder_output_tensor, mult_mask, mult_mask_sum, arr_D, decoder_input_with_dt = build_decoder_input_output_parallel(
    TrainingInput, TrainingOutput, TrainingMasks
)

val_decoder_input_tensor, val_decoder_output_tensor, val_mult_mask, val_mult_mask_sum, val_arr_D, val_decoder_input_with_dt  = build_decoder_input_output_parallel(
    ValidationInput, ValidationOutput, ValidationMasks
)
################################################################################################################################################

# MODEL :
################################################################################################################################################
input_dim = 2 * d_out + 1  # current + prev pulse

# OG TEST 2 : 
#model = MemoryUpdateLSTMWithAttention(input_dim=input_dim, hidden_dim=640, output_dim=d_out,num_layers=5, dropout=0.03894546817999088, pack_with_mask=True).to(device)

# TEST 3 (Optuna) : 
#model = MemoryUpdateLSTMWithAttention(input_dim=input_dim, hidden_dim=768, output_dim=d_out,num_layers=6, dropout=0.03630097401781031, pack_with_mask=True).to(device)

# TEST 4 : 
model = MemoryUpdateLSTMWithAttention(input_dim=input_dim, hidden_dim=640, output_dim=d_out,num_layers=5, dropout=0.09777095775579354, pack_with_mask=True).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=param["lr"], weight_decay=param["weight_decay"])

# LR SCHEDULER :
warmup_steps = int(param["NDataT"] / param["batch_size"]) * param["warmup"]
max_steps = int(param["NDataT"] / param["batch_size"]) * param["n_epochs"]
lr_scheduler = CosineScheduler(optimizer, warmup_steps, max_steps, max_lr=param["max_lr"])
################################################################################################################################################


# LOSS : (dataset-level normalized NRMSE)
################################################################################################################################################
# dimension weights (you already had this as [1,1,D]; we’ll keep it but use it as a flat [D] inside the loss)
loss_weights = torch.tensor([5.0, 2.0] + [0.2] * (d_out - 3) + [2.0]).view(1, 1, -1).to(device)

@torch.no_grad()
def _compute_dataset_stats(y, mask, eps=1e-6):
    """
    y:    [N, T, D]
    mask: [N, T]  (1 valid, 0 pad)
    returns mean_d [D], var_d [D] computed once on TRAIN set
    """
    m = mask.unsqueeze(-1).to(y.dtype)          # [N,T,1]
    N = m.sum() + eps                           # scalar masked count
    mean_d = (y * m).sum(dim=(0,1)) / N         # [D]
    var_d  = ((y - mean_d)**2 * m).sum(dim=(0,1)) / N
    var_d  = torch.clamp(var_d, min=1e-6)
    return mean_d, var_d

# Precompute fixed dataset stats on TRAIN targets (CPU tensors; moved to device inside loss)
_mean_d_fixed, _var_d_fixed = _compute_dataset_stats(decoder_output_tensor, mult_mask)

# OPTION 2 - LOSS 4 : 
def norm_weight_masked_rmse_loss(pred, target, mask, loss_weights=loss_weights, eps=1e-6):
    #assert pred.shape == target.shape, f"pred {pred.shape} vs target {target.shape}"
    #assert mask.shape[:2] == pred.shape[:2], f"mask {mask.shape} vs pred {pred.shape}"

    w = (loss_weights.view(-1).to(pred.device, pred.dtype))
    w = w / (w.sum() + eps)

    mean_d = _mean_d_fixed.to(pred.device, pred.dtype)
    var_d  = _var_d_fixed.to(pred.device,  pred.dtype)

    # masked MSE per dimension over the current batch :
    m  = mask.unsqueeze(-1).to(pred.dtype)                
    N  = m.sum() + eps                                   
    mse_d = (((pred - target)**2) * m).sum(dim=(0,1)) / N 

    # dataset-normalized MSE per dimension :
    nmse_d = (mse_d + eps) / (var_d + eps)

    # weighted average :
    return (nmse_d * w).sum()

""" @torch.no_grad()
def mean_predictor_baseline(y, mask, w):
    #  sanity check: predict per-dim dataset mean -> NRMSE ≈ 1.0
    B,T,D = y.shape
    mean_d = _mean_d_fixed.to(y.device, y.dtype)
    pred   = mean_d.view(1,1,D).expand(B,T,D)
    return norm_weight_masked_rmse_loss(pred, y, mask, w).item()

baseline = mean_predictor_baseline(decoder_output_tensor.to(device),
                                   mult_mask.to(device),
                                   torch.tensor([5.,2.]+[0.2]*(d_out-3)+[2.],
                                                device=device))
print(f"Mean-predictor NRMSE ≈ {baseline:.3f} (should be ~1.0)") """
################################################################################################################################################

# TRAINING LOOP 
################################################################################################################################################
TrainingError = []
Sqrt_TrainingError = []

ValidationError = []
Sqrt_ValidationError = []

global_step = 0
total_steps = (decoder_input_tensor.shape[0] // batch_size) * n_epochs

for epoch in tqdm(range(n_epochs)):
    model.train()
    total_loss, sqrt_total_loss = 0.0, 0.0

    n_batches = decoder_input_tensor.shape[0] // batch_size
    
    for i in range(n_batches):

        xb = decoder_input_with_dt[i*batch_size:(i+1)*batch_size].to(device)   # [B,T,2*d_out+1]
        yb = decoder_output_tensor[i*batch_size:(i+1)*batch_size].to(device)   # [B,T,d_out]
        maskb = mult_mask[i*batch_size:(i+1)*batch_size].to(device)               # [B,T]
        arrDb = arr_D[i*batch_size:(i+1)*batch_size].to(device)                   # [B,T]

        #p_tf = tf_ratio(global_step, total_steps, start=1.0, end=0.0, mode="linear")
        #p_tf = tf_ratio(global_step, total_steps, start=0.8515292433146034, end=0.24001717597446412, mode="exp")
        p_tf = tf_ratio(global_step, total_steps, start=0.9408256441291855, end=0.2651151319027647, mode="linear")
        
        start_token = torch.zeros(xb.size(0), d_out, device=device)

        optimizer.zero_grad()
        pred = scheduled_sampling_decode_lstm_luong_cached(model, xb, maskb, d_out, arrDb, p_tf, start_token)

        loss = norm_weight_masked_rmse_loss(pred, yb, maskb)
        loss.backward()
        optimizer.step()
        if lr_scheduler: lr_scheduler.step()

        total_loss += loss.item()
        sqrt_total_loss += math.sqrt(loss.item())
        global_step += 1

    TrainingError.append(total_loss / n_batches)
    Sqrt_TrainingError.append(sqrt_total_loss / n_batches)

    #  AR validation 
    model.eval()
    with torch.no_grad():
        xb_val   = val_decoder_input_with_dt.to(device)
        yb_val   = val_decoder_output_tensor.to(device)
        mask_val = val_mult_mask.to(device)
        arrD_val = val_arr_D.to(device)
        start_token = torch.zeros(xb_val.size(0), d_out, device=device)

        pred_val = ar_decode_lstm_luong_cached(model, xb_val, mask_val, d_out, start_token, arr_D=arrD_val)

        loss_val = norm_weight_masked_rmse_loss(pred_val, yb_val, mask_val)
        sqrt_loss_val = math.sqrt(loss_val.item())

        ValidationError.append(loss_val.item())
        Sqrt_ValidationError.append(sqrt_loss_val)

    print(f"Epoch {epoch+1}/{n_epochs} - RMSE: {TrainingError[-1]:.4f}, √Loss: {Sqrt_TrainingError[-1]:.4f}")
    print(f"Validation - RMSE: {loss_val:.4f}, √Loss: {sqrt_loss_val:.4f}")


# LOSS PLOT : 
################################################################################################################################################
title_graph = f"{Current_Model} - SQRT Training vs Validation Loss"
plot_graph_with_min(title_graph, graph_path, sqrt_loss_graph_file, Sqrt_TrainingError, Sqrt_ValidationError)

# VISUALIZAITON : 
################################################################################################################################################
model.eval()
with torch.no_grad(): 
    for sample_idx in [0, 3, 7, 10] :

        # [1] PREDICTIONS ON VALIDATION DATA :
        #val_pred = model(val_decoder_input_tensor.to(device))  # [B, T, d_out]
        start_token = torch.zeros(val_decoder_input_with_dt.size(0), d_out, device=device)

        #val_pred = autoregressive_decode_from_flat_prefix(model, val_decoder_input_with_dt.to(device), val_mult_mask.to(device), d_out, start_token)
        val_pred = ar_decode_lstm_luong_cached(
            model, val_decoder_input_with_dt.to(device), val_mult_mask.to(device),
            d_out, start_token, arr_D=val_arr_D.to(device)
        )
        pred_sample = val_pred[sample_idx]       # [T, d_out]
        
        #print(f"Sample {sample_idx}")
        #print(f"Prediction shape: {pred_sample.shape}")
        #print("Prediction Sample : ", pred_sample)

        # [2] ARRAY D - array to mask valid outputs :
        val_arr_D = val_arr_D.to(device)  # [B, T]
        arr_d_sample = val_arr_D[sample_idx]     # [T]

        #print("\nArray D: ", arr_d_sample) 

        # [3] CUT PREDICTION - keep only valid pulses + remove padding :
        val_mult_mask = val_mult_mask.to(device)  # [B, T]
        Length_int = int(val_mult_mask[sample_idx].sum().item())  # ignore end token        
        Cut_Prediction = pred_sample[:Length_int-1]  # [T_out_valid, d_out]
       
        #print(f"Length of Val Mult Mask : {Length_int}")
        #print("Cut Prediction w/ until (length - 1) : ", Cut_Prediction)  

        Cut_arr_D = arr_d_sample.cpu().tolist()  # [T]
        Cut_arr_D = Cut_arr_D[:Length_int]  # [T_out_valid]
        Cut_arr_D = torch.tensor(Cut_arr_D, device=Cut_Prediction.device)  # now [T_out_valid]

        #print("Cut arr D until length : ", Cut_arr_D) 
        
        # remove first element from cut arr D 
        Cut_arr_D = Cut_arr_D[1:]

        #print("Cut arr D after removing first element - 0 : ", Cut_arr_D) 

        Cut_Prediction = Cut_Prediction[Cut_arr_D == 1].cpu().tolist()

        #print("Cut Prediction w/ arr D - 0 : ", Cut_Prediction) 

        # [4] InputData = original pulse sequence (before decoding)
        InputData = ValidationInput[sample_idx].cpu().tolist()

        #print("Input Data : ", InputData)
        
        # [5] OutputData = ground truth output sequence with padding
        OutputData = ValidationOutput[sample_idx].cpu().tolist()

        #print("Output Data : ", OutputData)

        # Remove padding from ground truth output using (OG) mult_mask
        OG_mult_mask = ValidationMasks[1].to(device)  # [B, T]
        Length_int = int(OG_mult_mask[sample_idx].sum().item() - 1)  # ignore end token
        #print(f"Length of the sample {sample_idx} : {Length_int}")
        Cut_Output = OutputData[:Length_int]  # [T_out_valid, d_out]

        #print("Cut Output - 0 : ", Cut_Output)

        # VISUALIZATION OF THE PREDICTIONS :
        plot_file = f"{visuals_plot_file}_sample_{sample_idx}.png"
        plot_visuals(InputData, Cut_Output, Cut_Prediction, plot_file, visuals_path)
################################################################################################################################################