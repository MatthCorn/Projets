import torch
import torch.nn as nn
from Complete.Transformer.EasyFeedForward import FeedForward

class RNNEncoder(nn.Module):
    def __init__(self, n_encoder, d_in=10, d_att=128, WidthsEmbedding=[32], dropout=0.):
        super(RNNEncoder, self).__init__()
        self.hidden_dim = d_att
        self.num_layers = n_encoder

        self.Embedding = FeedForward(d_in=d_in, d_out=d_att, widths=WidthsEmbedding, dropout=dropout)
        self.Decoding = FeedForward(d_in=2*d_att, d_out=d_in, widths=WidthsEmbedding, dropout=dropout)

        # RNN layer (can be LSTM or GRU)
        self.rnn = nn.LSTM(d_att, d_att, num_layers=n_encoder, bidirectional=True, batch_first=True)

        self.register_buffer('h_0', torch.zeros(self.num_layers * 2, 1, self.hidden_dim))
        self.register_buffer('c_0', torch.zeros(self.num_layers * 2, 1, self.hidden_dim))

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, d_in)
        x = self.Embedding(x)

        # Initialize hidden state and cell state
        if self.h_0.size(1) != x.size(0):
            self.register_buffer('h_0', torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim, device=x.device))
            self.register_buffer('c_0', torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim, device=x.device))

        # Pass through RNN
        output, _ = self.rnn(x, (self.h_0, self.c_0))

        # Output shape: (batch_size, sequence_length, hidden_dim * num_directions) = (batch_size, sequence_length, channels)
        return self.Decoding(output)

# Example usage
if __name__ == "__main__":
    batch_size = 16
    seq_len = 100
    input_dim = 10
    hidden_dim = 128

    # Example input
    x = torch.randn(batch_size, seq_len, input_dim)  # (batch_size, channels, sequence_length)

    # RNN Encoder
    rnn_encoder = RNNEncoder(n_encoder=3, d_in=input_dim, d_att=hidden_dim)
    output = rnn_encoder(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
