from Complete.Transformer.EncoderTransformer import EncoderLayer
from Complete.Transformer.PositionalEncoding import PositionalEncoding
from Complete.Transformer.EasyFeedForward import FeedForward
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, n_encoder, len_in, d_in=10, d_att=64, n_heads=4, WidthsEmbedding=[32], norm='post'):
        super().__init__()
        self.PEIn = PositionalEncoding(d_att=d_att, dropout=0, max_len=len_in)

        self.Encoders = nn.ModuleList()
        for i in range(n_encoder):
            self.Encoders.append(EncoderLayer(d_att=d_att, n_heads=n_heads, norm=norm))

        self.Embedding = FeedForward(d_in=d_in, d_out=d_att, widths=WidthsEmbedding, dropout=0)

        self.Classifier = FeedForward(d_in=d_att, d_out=len_in + 1, widths=WidthsEmbedding, dropout=0)

    def forward(self, x):
        x = self.Embedding(x)
        x = self.PEIn(x)
        for Encoder in self.Encoders:
            x = Encoder(x)

        x = self.Classifier(x)
        x = F.softmax(x, dim=-1)

        return x



