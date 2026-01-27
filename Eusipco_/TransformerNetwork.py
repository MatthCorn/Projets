from Complete.Transformer.EncoderTransformer import EncoderLayer
from Complete.Transformer.PositionalEncoding import PositionalEncoding
from Complete.Transformer.EasyFeedForward import FeedForward
import torch.nn as nn

class Network(nn.Module):
    def __init__(
            self,
            n_layers,
            len_in=10,
            d_in=10,
            d_model=128,
            n_heads=4,
            WidthsEmbedding=[32],
            norm='post',
            dropout=0,
            **kwargs
    ):
        super().__init__()
        self.PosEncoding = PositionalEncoding(d_att=d_model, dropout=dropout, max_len=len_in)
        self.Encoders = nn.ModuleList()
        for i in range(n_layers):
            self.Encoders.append(EncoderLayer(d_att=d_model, n_heads=n_heads, norm=norm, dropout_FF=dropout, dropout_SA=dropout))
        self.Embedding = FeedForward(d_in=d_in, d_out=d_model, widths=WidthsEmbedding, dropout=dropout)
        self.Decoding = FeedForward(d_in=d_model, d_out=d_in, widths=WidthsEmbedding, dropout=dropout)

    def forward(self, x):
        x = self.Embedding(x)
        x = self.PosEncoding(x)
        for Encoder in self.Encoders:
            x = Encoder(x)
        return self.Decoding(x)

