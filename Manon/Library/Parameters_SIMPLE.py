# PARAMETERS : 
################################################################################################################################################
def get_parameter() : 
    return {
    "len_in": 10,
    "len_out": 20,
    "d_in": 10,
    "n_pulse_plateau": 5,
    "sensitivity": 0.1,
    "lr": 0.00003, # 0.0001 # 0.0003 # 1e-4
    "lr_option": {
        "value": 0.00003, # 0.0001 # 0.0003
        "reset": "y",
        "type": "cos"
    },
    # what is this for : 
    "mult_grad": 1000,
    "weight_decay": 1e-3,
    "NDataT": 1000000,
    "NDataV": 1000,
    # how much iterations / steps per epoch / training does this equal
    "batch_size": 100,
    "n_epochs": 80,
    "distrib": "log",
    "error_weighting": "y",
    "max_lr": 5,
    "warmup": 5,
    "training_strategy": [
        {"mean": [-5, 5], "std": [0.2, 1]},
    ]
    }
################################################################################################################################################


