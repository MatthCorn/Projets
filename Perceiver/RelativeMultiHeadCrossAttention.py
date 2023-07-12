import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# Dans cette version, on tente de rendre RLCA plus souple que dans RelativeMultiHeadCrossAttentionV0 en autorisant x_latent à avoir des
# longueurs de séquence variable (mais bornée) entre batch. Il fera dans l'idée extactement les mêmes calculs que
# RelativeMultiHeadCrossAttentionV0 si cette longueur ne varie pas entre batchs.

#Relative Latent Cross Attention
class RLCA(nn.Module):
    def __init__(self, d_latent, d_input, d_att, num_heads, RPR_len=16, latent_len=32, max_len=64, dropout=0.1, masked=True, relative=True):
        super().__init__()
        d_head, remainder = divmod(d_att, num_heads)
        if remainder:
            raise ValueError("incompatible `d_att` and `num_heads`")

        self.RPR_len = RPR_len
        self.d_att = d_att
        self.d_latent = d_latent
        self.d_input = d_input
        self.d_head = d_head
        self.num_heads = num_heads
        self.key = nn.Linear(d_input, d_att)
        self.value = nn.Linear(d_input, d_att)
        self.query = nn.Linear(d_latent, d_att)
        self.finalLinear = nn.Linear(d_att, d_latent)
        self.dropout = nn.Dropout(dropout)
        self.relative = relative
        if self.relative:
            self.Er = nn.Parameter(torch.randn(2*RPR_len-1, d_head))
        self.masked = masked
        if self.masked:
            self.register_buffer("mask", self.MakeMask(latent_len=latent_len, input_len=max_len).unsqueeze(0).unsqueeze(0))
            # self.mask.shape = (1, 1, latent_len, max_len)
        self.previousInputLen = max_len
        self.previousLatentLen = latent_len
        self.register_buffer("ConvDistrib", self.Weight(M=max_len, N=latent_len, sigma=1.5))

    def forward(self, x_latent, x_input):
        # x_input.shape = (batch_size, input_len, d_input)
        batch_size, input_len, _ = x_input.shape
        # x_latent.shape = (batch_size, latent_len, d_latent)
        # OR
        # x_latent.shape = (1, latent_len, d_latent)
        _, latent_len, d_latent = x_latent.shape
        Kt = self.key(x_input).reshape(batch_size, input_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # Kt.shape = (batch_size, num_heads, d_head, input_len)
        V = self.value(x_input).reshape(batch_size, input_len, self.num_heads, -1).transpose(1, 2)
        # V.shape = (batch_size, num_heads, input_len, d_head)
        Q = self.query(x_latent).reshape(-1, latent_len, self.num_heads, self.d_head).transpose(1, 2)
        # Q.shape = (batch_size, num_heads, latent_len, d_head)
        # OR
        # Q.shape = (1, num_heads, latent_len, d_head)

        mask = None
        Ert = None
        if self.previousInputLen != input_len or self.previousLatentLen != latent_len:
            if self.masked:
                self.mask = self.MakeMask(latent_len=latent_len, input_len=input_len).unsqueeze(0).unsqueeze(0).to(self.Er.device, self.Er.dtype)
                self.previousInputLen = input_len
                self.previousLatentLen = latent_len
                mask = self.mask
                # mask.shape = (1, 1, latent_len, input_len)
            if self.relative:
                self.ConvDistrib = self.Weight(M=input_len, N=latent_len, sigma=1.5).to(self.Er.device, self.Er.dtype)
                # ConvDistrib.shape = (latent_len, 2*RPR_len-1, input_len)
                Ert = self.Er.transpose(0, 1)
                # Ert.shape = (d_head, 2*latent_len-1)
        else:
            if self.masked:
                mask = self.mask
                # mask.shape = (1, 1, latent_len, input_len)
            if self.relative:
                Ert = self.Er.transpose(0, 1)
                # Ert.shape = (d_head, 2*latent_len-1)


        RCA = self.RelativeCrossAttention(Q, Kt, Ert, V, input_len, Mask=mask)
        # RCA.shape = (batch_size, num_heads, latent_len, d_head)

        Concat = RCA.transpose(1, 2).reshape(batch_size, latent_len, -1)
        # Concat.shape = (batch_size, latent_len, d_att)
        out = self.finalLinear(Concat)
        # out.shape = (batch_size, seq_len, d_latent)
        return self.dropout(out)

    def RelativeCrossAttention(self, Q, Kt, Ert, V, input_len, Mask=None):
        # Q.shape = (batch_size, num_heads, latent_len, d_head)
        # or
        # Q.shape = (1, num_heads, latent_len, d_head)
        _, _, latent_len, _ = Q.shape

        Dh = self.d_head

        QKt = torch.matmul(Q, Kt)
        # QKt.shape = (batch_size, num_heads, latent_len, input_len)

        if Ert is not None:
            # Ert.shape = (d_head, 2*RPR_len-1)
            QEr = torch.matmul(Q, Ert)
            # QEr.shape = (batch_size, num_heads, latent_len, 2*RPR_len-1)
            # or
            # QEr.shape = (1, num_heads, latent_len, 2*RPR_len-1)
            # V.shape = (batch_size, num_heads, input_len, d_head)
            # ConvDistrib.shape = (latent_len, 2*RPR_len-1, input_len)

            QEr = QEr.unsqueeze(dim=-2)
            # QEr.shape = (batch_size, num_heads, latent_len, 1, 2*RPR_len-1)
            # OR
            # QEr.shape = (1, num_heads, latent_len, 1, 2*RPR_len-1)

            # Here we want the i-th vector of QEr (shape = (1, 2*RPR_len-1)) to be multiplied with the i-th matrix of ConvDistrib (shape = (2*RPR_len-1, input_len)),
            # i in [1; latent_len], in order to get a matrix of the shape (latent_len, input_len)

            Srel = torch.matmul(QEr, self.ConvDistrib)
            # Srel.shape = (batch_size, num_heads, latent_len, 1, input_len)
            # OR
            # Srel.shape = (1, num_heads, latent_len, 1, input_len)
            Srel = Srel.squeeze(dim=-2)
            # Srel.shape = (batch_size, num_heads, latent_len, input_len)
            # OR
            # Srel.shape = (1, num_heads, latent_len, input_len)


            Attention = (QKt + Srel) / math.sqrt(Dh)
        else:
            Attention = QKt / math.sqrt(Dh)
        # Attention.shape = (batch_size, num_heads, latent_len, input_len)
        if Mask is not None:
            # Mask.shape = (1, 1, latent_len, input_len)
            Attention = Attention.masked_fill(Mask == 0, float("-inf"))
        Attention = F.softmax(Attention, dim=-1)
        out = torch.matmul(Attention, V)
        # out.shape = (batch_size, num_heads, latent_len, d_head)
        return out


    def MakeMask(self, latent_len, input_len):
        N = latent_len
        M = input_len
        mat = torch.zeros((N, M))
        for i in range(N):
            for j in range(M):
                mat[i, j] = i/N >= j/M
        return mat

    def Weight(self, M, N, sigma):
        mat = torch.zeros((N, 2*self.RPR_len-1, M))
        for i in range(N):
            for j in range(M):
                for k in range(2*self.RPR_len-1):
                    alpha = (2*self.RPR_len-1) * (i/N - j/M + 1)/2
                    mat[i, k, j] = -(k-alpha)**2/sigma
        mat = torch.softmax(mat, dim=1)
        return mat
