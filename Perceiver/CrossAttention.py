import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def Weight(N, M, sigma):
    mat = torch.zeros((N,2*N-1,M))
    for i in range(N):
        for j in range(M):
            for k in range(2*N-1):
                mat[i,k,j] = -(k-(i-j*N/M))**2/sigma
    mat = torch.softmax(mat, dim=1)
    return mat

#Relative Latent Cross Attention
class RLCA(nn.Module):
    def __init__(self, d_model, num_heads, d_latent=32, max_len=64, dropout=0.1, masked=True):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.d_latent = d_latent
        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.finalLinear = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(2*d_latent-1, d_head))
        self.masked = masked
        if self.masked:
            self.register_buffer(
                "mask",
                self.MakeMask(d_latent, max_len)
                .unsqueeze(0).unsqueeze(0)
            )
            # self.mask.shape = (1, 1, d_latent, max_len)

    def forward(self,x_Latent,x_Input):
        # x_Input.shape = (batch_size, input_len, d_model)
        batch_size, input_len, _ = x_Input.shape
        # x_Latent.shape = (batch_size, latent_len, d_model)
        _, latent_len, _ = x_Latent.shape
        Kt = self.key(x_Latent).reshape(batch_size, input_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # Kt.shape = (batch_size, num_heads, d_head, input_len)
        V = self.value(x_Input).reshape(batch_size, input_len, self.num_heads, -1).transpose(1, 2)
        # V.shape = (batch_size, num_heads, input_len, d_head)
        Q = self.query(x_Input).reshape(batch_size, latent_len, self.num_heads, -1).transpose(1, 2)
        # Q.shape = (batch_size, num_heads, latent_len, d_head)

        start = self.d_latent - 1 - (latent_len-1)
        finish = self.d_latent - 1 + latent_len
        Ert = self.Er[start:finish, :].transpose(0, 1)
        # Ert.shape = (d_head, 2*latent_len-1)
        if self.masked:
            mask = self.mask[:, :, :latent_len, :latent_len]
            # mask.shape = (1, 1, latent_len, latent_len)
            RSA = self.RelativeSelfAttention(Q, Kt, Ert, self.d_head, V, latent_len, input_len,Mask=mask)
        else:
            RSA = self.RelativeSelfAttention(Q, Kt, Ert, self.d_head, V, latent_len, input_len)
        # RSA.shape = (batch_size, num_heads, latent_len, d_head)

        Concat = RSA.transpose(1,2).reshape(batch_size,latent_len,-1)
        # Concat.shape = (batch_size, latent_len, d_model)
        out = self.finalLinear(Concat)
        # out.shape = (batch_size, latent_len, d_model)
        return self.dropout(out)

    def RelativeSelfAttention(self,Q, Kt, Ert, Dh, V, latent_len, input_len Mask=None):
        # Ert.shape = (d_head, 2*latent_len-1)
        QEr = torch.matmul(Q, Ert)
        # QEr.shape = (batch_size, num_heads, latent_len, 2*latent_len-1)
        # V.shape = (batch_size, num_heads, input_len, d_head)

        ConvDistrib = Weight(N=latent_len, M=input_len, sigma=1.5)
        # ConvDistrib.shape = (latent_len, 2*latent_len-1, input_len)

        QEr = QEr.unsqueeze(dim=-2)
        # QEr.shape = (batch_size, num_heads, latent_len, 1, 2*latent_len-1)

        # Here we want the i-th vector of QEr (shape = (1, 2*latent_len-1)) to be multiplied with the i-th matrix of ConvDistrib (shape = (2*latent_len-1, input_len)),
        # i in [1; latent_len], in order to get a matrix of the shape (latent_len, input_len)

        Srel = torch.matmul(QEr,ConvDistrib)
        # Srel.shape = (batch_size, num_heads, latent_len, 1, input_len)
        Srel = Srel.squeeze(dim=-2)
        # Srel.shape = (batch_size, num_heads, latent_len, input_len)

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

    def Make_Mask(self,)