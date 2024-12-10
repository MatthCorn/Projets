from PointTransformer.TSelfAttentionEval import SA as TSA
from PointTransformer.SelfAttention import SA as PSA
from PointTransformer.OpSelfAttentionEval import SA as OSA
import time
import torch

device = torch.device('cpu')

point = True

d_att = 64
n_head = 4
d_group = 16
batch_size = 10000
len_seq = 25


# N = PSA(d_att, d_group).to(device)
# N = GSA(d_att, d_group).to(device)
N = OSA(d_att, d_group).to(device)
# N = TSA(d_att).to(device)

print('création SA')

print(sum(p.numel() for p in N.parameters()))
time.sleep(5)

x = torch.normal(0, 1, (batch_size, len_seq, d_att)).to(device)
y = N(x)
print('calcul sortie')
time.sleep(5)


