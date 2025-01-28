import torch
import torch.nn as nn
import torch.nn.functional as F
from Complete.Transformer.EasyFeedForward import FeedForward

class ResidualBlock(nn.Module):
    def __init__(self, d_att, size=3, rate=1, is_first=False):
        super(ResidualBlock, self).__init__()
        self.is_first = is_first
        self.rate = rate

        # Conv layers
        self.conv_in = nn.Conv1d(d_att, d_att // 2, kernel_size=1)
        self.a_conv = nn.Conv1d(d_att // 2, d_att // 2, kernel_size=size, dilation=rate, padding=((size - 1) * rate) // 2)
        self.conv_out = nn.Conv1d(d_att // 2, d_att, kernel_size=1)

        # Layer normalization
        self.ln = nn.LayerNorm(d_att // 2) if not is_first else None

    def forward(self, x):
        residual = x

        # First convolution with ReLU activation
        x = F.relu(self.conv_in(x))

        # Layer normalization if not the first block
        if self.ln is not None:
            x = self.ln(x.transpose(1, 2)).transpose(1, 2)

        x = F.relu(self.a_conv(x))

        # Output convolution and residual connection
        x = self.conv_out(x)
        return x + residual

class Encoder(nn.Module):
    def __init__(self, n_encoder, d_in=10, d_att=128, WidthsEmbedding=[32], dropout=0.):
        super(Encoder, self).__init__()
        self.n_encoder = n_encoder

        self.Embedding = FeedForward(d_in=d_in, d_out=d_att, widths=WidthsEmbedding, dropout=dropout)
        self.Decoding = FeedForward(d_in=d_att, d_out=d_in, widths=WidthsEmbedding, dropout=dropout)

        # Create residual blocks
        self.blocks = nn.ModuleList()
        for i in range(n_encoder):
            self.blocks.append(ResidualBlock(d_att, size=5, rate=1, is_first=True))
            self.blocks.append(ResidualBlock(d_att, size=5, rate=2))
            self.blocks.append(ResidualBlock(d_att, size=5, rate=4))
            self.blocks.append(ResidualBlock(d_att, size=5, rate=8))
            self.blocks.append(ResidualBlock(d_att, size=5, rate=16))

    def forward(self, x):
        x = self.Embedding(x).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.Decoding(x.transpose(1, 2))

# Example usage
if __name__ == "__main__":
    batch_size = 16
    seq_len = 100
    in_dim = 10

    # Example input
    x = torch.randn(batch_size, seq_len, in_dim)  # (batch_size, sequence_length, channels)

    # Encoder
    encoder = Encoder(n_encoder=3, d_in=in_dim)
    output = encoder(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
