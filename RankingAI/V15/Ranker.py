from Complete.Transformer.DecoderTransformer import DecoderLayer as EncoderLayer
from Complete.Transformer.PositionalEncoding import PositionalEncoding
from Complete.Transformer.EasyFeedForward import FeedForward
from Complete.Transformer.LearnableModule import LearnableParameters
import torch.nn as nn
import torch

class Network(nn.Module):
    def __init__(self, n_encoder, len_in, len_latent, d_in=10, d_att=64, n_heads=4, WidthsEmbedding=[32], norm='post'):
        super().__init__()
        self.PEIn = PositionalEncoding(d_att=d_att, dropout=0, max_len=len_in)

        self.Latent = LearnableParameters(torch.normal(mean=torch.zeros(1, len_latent, d_att)))

        self.Encoders = nn.ModuleList()
        for i in range(n_encoder):
            self.Encoders.append(EncoderLayer(d_att=d_att, n_heads=n_heads, norm=norm))

        self.Embedding = FeedForward(d_in=d_in, d_out=d_att, widths=WidthsEmbedding, dropout=0)

        self.Decoding = FeedForward(d_in=d_att, d_out=d_in, widths=WidthsEmbedding, dropout=0)

    def forward(self, x):
        x = self.Embedding(x)
        x = self.PEIn(x)
        y = self.Latent()
        for Encoder in self.Encoders:
            y = Encoder(y, x)
        return self.Decoding(y)
