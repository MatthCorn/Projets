import torch
import numpy as np
import torch.nn as nn

def GetType(module):
    return str(type(module)).split("'")[1].split(".")[-1]

def T_Fixup(module, location='origin'):
    for name, module in module.named_childrent():
        if 'embedding' in name:
            for linear in module.linears:
                nn.init.normal_(linear.data)
        if GetType(module) == 'LearnableParameters':
            nn.init.normal_(module.param.data)
        if 'encoders' in name:
            T_Fixup(module, 'encoder')
        if 'decoders' in name:
            T_Fixup(module, 'decoder')
        if GetType(module) == 'FeedForward':
            for linear in module.linears:
                nn.init.xavier_uniform_(linear.data)
        if GetType(module) in ['MHCA', 'MHSA']:
            nn.init.xavier_uniform_(module.key.weight)
            nn.init.xavier_uniform_(module.value.weight)
            nn.init.xavier_uniform_(module.query.weight)
            nn.init.xavier_uniform_(module.linear.weight)
            if location == 'encoder':
                module.value.weight /= 0.67
                module.linear.weight /= 3
            if location == 'decoder':
                module.value.weight /= 0.2
                module.linear.weight /= 34
