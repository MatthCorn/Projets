from Eusipco_.ConvolutionBlock import EncoderLayer
from Complete.Transformer.EasyFeedForward import FeedForward
import torch.nn as nn

class Network(nn.Module):
    def __init__(
            self,
            n_layers,
            d_in=10,
            d_model=128,
            kernel_size=4,
            WidthsEmbedding=[32],
            norm='post',
            dropout=0,
            **kwargs
    ):
        super().__init__()
        self.Encoders = nn.ModuleList()
        for i in range(n_layers):
            self.Encoders.append(EncoderLayer(d_model=d_model, kernel_size=kernel_size, norm=norm, dropout_FF=dropout, dropout_Conv=dropout))
        self.Embedding = FeedForward(d_in=d_in, d_out=d_model, widths=WidthsEmbedding, dropout=dropout)
        self.Decoding = FeedForward(d_in=d_model, d_out=d_in, widths=WidthsEmbedding, dropout=dropout)

    def forward(self, x):
        x = self.Embedding(x)
        for Encoder in self.Encoders:
            x = Encoder(x)
        return self.Decoding(x)

