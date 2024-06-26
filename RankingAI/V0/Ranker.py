from Complete.Transformer.EncoderTransformer import EncoderLayer
from Complete.Transformer.PositionalEncoding import PositionalEncoding
from Complete.Transformer.EasyFeedForward import FeedForward
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, n_encoder, max_len, d_in=10, d_att=64, n_heads=4, WidthsEmbedding=[32]):
        super().__init__()
        self.PosEncoding = PositionalEncoding(d_att=d_att, dropout=0, max_len=max_len)
        self.Encoders = nn.ModuleList()
        for i in range(n_encoder):
            self.Encoders.append(EncoderLayer(d_att=d_att, n_heads=n_heads, norm='pre'))
        self.Embedding = FeedForward(d_in=d_in, d_out=d_att, widths=WidthsEmbedding, dropout=0)
        self.Classifier = FeedForward(d_in=d_att, d_out=1, widths=[32, 16], dropout=0)

    def forward(self, x):
        x = self.Embedding(x)
        x = self.PosEncoding(x)
        for Encoder in self.Encoders:
            x = Encoder(x)
        return self.Classifier(x)



