from PointTransformer.TSelfAttentionEval import SA as TSA
from PointTransformer.PSelfAttentionEval import SA as PSA
import time
import torch

device = torch.device('cpu')

point = True

d_att = 64
batch_size = 10000
len_seq = 25

if point:
    N = PSA(d_att).to(device)
    print('création point SA')

else:
    N = TSA(d_att).to(device)
    print('création classic SA')

print(sum(p.numel() for p in N.parameters()))
time.sleep(1)

x = torch.normal(0, 1, (batch_size, len_seq, d_att)).to(device)
y = N(x)
print('calcul sortie')
time.sleep(1)


