import torch
import math
from torch import nn
import torch.nn.functional as F
import time

#Relative Multi-Head Self Attention
class RMHSA(nn.Module):
    def __init__(self, d_model, num_heads, max_len=64, dropout=0.1, masked=True):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.finalLinear = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(2*max_len-1, d_head))
        self.masked = masked
        if self.masked:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(max_len, max_len))
                .unsqueeze(0).unsqueeze(0)
            )
            # self.mask.shape = (1, 1, max_len, max_len)
        self.InferenceTime = {'KVQ':0,'Ert':0,'RSA':0}

    def forward(self,x):
        # x.shape = (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        t = time.time()
        Kt = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # Kt.shape = (batch_size, num_heads, d_head, seq_len)
        V = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # V.shape = (batch_size, num_heads, seq_len, d_head)
        Q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # Q.shape = (batch_size, num_heads, seq_len, d_head)
        self.InferenceTime['KVQ'] += time.time()-t

        t = time.time()
        start = self.max_len - 1 - (seq_len-1)
        finish = self.max_len - 1 + seq_len
        Ert = self.Er[start:finish, :].transpose(0, 1)
        self.InferenceTime['Ert'] += time.time() - t
        # Ert.shape = (d_head, 2*seq_len-1)
        t = time.time()
        if self.masked:
            mask = self.mask[:, :, :seq_len, :seq_len]
            # mask.shape = (1, 1, seq_len, seq_len)
            RSA = self.RelativeSelfAttention(Q, Kt, Ert, self.d_head, V, Mask=mask)
        else:
            RSA = self.RelativeSelfAttention(Q, Kt, Ert, self.d_head, V)
        # RSA.shape = (batch_size, num_heads, seq_len, d_head)
        self.InferenceTime['RSA'] += time.time()-t

        Concat = RSA.transpose(1,2).reshape(batch_size,seq_len,-1)
        # Concat.shape = (batch_size, seq_len, d_model)
        out = self.finalLinear(Concat)
        # out.shape = (batch_size, seq_len, d_model)
        return self.dropout(out)

    # This is much more efficient than PersonalSkew when dimensions go numberous and/or high
    def NotSkew(self,QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, 2*seq_len-1)
        padded_1 = F.pad(QEr, (0, 1))
        # padded_1.shape = (batch_size, num_heads, seq_len, 2*seq_len)
        batch_size, num_heads, seq_len, _ = padded_1.shape
        reshaped_1 = padded_1.reshape(batch_size, num_heads, -1)
        # reshaped_1.shape = (batch_size, num_heads, 2*seq_len**2)
        padded_2 = F.pad(reshaped_1,(0,seq_len-1))
        # padded_2.shape = (batch_size, num_heads, 2*seq_len**2+seq_len-1) = (batch_size, num_heads, (seq_len+1)*(2*seq_len-1))
        reshaped_2 = padded_2.reshape(batch_size, num_heads, seq_len+1, 2*seq_len-1)
        Srel = reshaped_2[:, :,:seq_len][:,:,:,-seq_len:]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel

    def PersonalSkew(self,QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, 2*seq_len-1)
        batch_size, num_heads, seq_len, _ = QEr.shape
        padded_1 = F.pad(QEr, (0,0,0,seq_len-1))
        # padded_1.shape = (batch_size, num_heads, 2*seq_len-1, 2*seq_len-1)
        padded_2 = F.pad(padded_1,(0,1))
        # padded_2.shape = (batch_size, num_heads, 2*seq_len-1, 2*seq_len)
        reshaped = padded_2.reshape(batch_size, num_heads, 2*seq_len, 2*seq_len-1)
        Srel = reshaped[:, :, :seq_len][:, :, :, -seq_len:]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel

    def RelativeSelfAttention(self,Q, Kt, Ert, Dh, V, Mask=None):
        # Ert.shape = (d_head, seq_len)
        QEr = torch.matmul(Q, Ert)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.NotSkew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)

        QKt = torch.matmul(Q, Kt)
        # QKt.shape = (batch_size, num_heads, seq_len, seq_len)
        Attention = (QKt + Srel) / math.sqrt(Dh)
        if Mask is not None:
            # Mask.shape = (1, 1, seq_len, seq_len)
            Attention = Attention.masked_fill(Mask == 0, float("-inf"))
        # Attention.shape = (batch_size, num_heads, seq_len, seq_len)
        Attention = F.softmax(Attention, dim=-1)
        out = torch.matmul(Attention, V)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        return out

# N = 1000
# QEr = torch.tensor([list(range(1,N*(2*N-1)+1))]).reshape(N,-1).unsqueeze(0).unsqueeze(0)
# import time
# t = time.time()
# for i in range(10000):
#     a = PersonalSkew(1,QEr)
# print(time.time() -t)
# t = time.time()
# for i in range(10000):
#     a = NotSkew(1,QEr)
# print(time.time() -t)