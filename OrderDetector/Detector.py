from Transformer.EncoderTransformer import EncoderLayer
from Transformer.EasyFeedForward import FeedForward
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, n_encoder, d_in=10, d_model=64, d_att=64, num_heads=4, relative=False, masked=False):
        super().__init__()
        self.Encoders = nn.ModuleList()
        for i in range(n_encoder):
            self.Encoders.append(EncoderLayer(d_model=d_model, d_att=d_att, num_heads=num_heads, WidthsFeedForward=[128],
                                              max_len=5, MHADropout=0, FFDropout=0, masked=masked, relative=relative))
        self.Embedding = FeedForward(d_in=d_in, d_out=d_model, widths=[32], dropout=0)
        self.Classifier = FeedForward(d_in=d_model, d_out=1, widths=[64, 32, 16], dropout=0)

    def forward(self, x):
        x = self.Embedding(x)
        for Encoder in self.Encoders:
            x = Encoder(x)
        y = self.Classifier(x)
        return y

