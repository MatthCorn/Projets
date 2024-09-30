from Complete.Transformer.EncoderTransformer import EncoderLayer
from Complete.Transformer.PositionalEncoding import PositionalEncoding
from Complete.Transformer.EasyFeedForward import FeedForward
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, n_encoder, max_len, d_in=10, d_att=64, n_heads=4, WidthsEmbedding=[32], PE=False, norm='pre'):
        super().__init__()

        self.Embedding = FeedForward(d_in=d_in, d_out=d_att, widths=[])

        if PE:
            self.PosEncoding = PositionalEncoding(d_att=d_att, dropout=0, max_len=max_len)
        else:
            self.PosEncoding = lambda x: x

        self.Encoders = nn.ModuleList()
        for i in range(n_encoder):
            self.Encoders.append(EncoderLayer(d_att=d_att, n_heads=n_heads, norm=norm))

        self.Classifier = FeedForward(d_in=d_att*max_len, d_out=1, widths=[64, 32, 8])

    def forward(self, x):
        batch_size = len(x)

        x = self.Embedding(x)
        x = self.PosEncoding(x)
        for Encoder in self.Encoders:
            x = Encoder(x)

        x = x.reshape(batch_size, -1)
        return self.Classifier(x)
