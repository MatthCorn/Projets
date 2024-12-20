from Complete.Transformer.DecoderTransformer import DecoderLayer as EncoderLayer
from Complete.Transformer.PositionalEncoding import PositionalEncoding
from Complete.Transformer.EasyFeedForward import FeedForward
from Complete.Transformer.LearnableModule import LearnableParameters
import torch.nn as nn
import torch
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, n_encoder, len_in, len_latent, d_in=10, d_att=64, n_heads=4, WidthsEmbedding=[32], norm='post', dropout=0):
        super().__init__()
        self.PEIn = PositionalEncoding(d_att=d_att, dropout=dropout, max_len=len_in)

        self.Latent = LearnableParameters(torch.normal(0, 1, (1, len_latent, d_att)))

        self.Encoders = nn.ModuleList()
        for i in range(n_encoder):
            self.Encoders.append(EncoderLayer(d_att=d_att, n_heads=n_heads, norm=norm, dropout_FF=dropout, dropout_SA=dropout))

        self.Embedding = FeedForward(d_in=d_in, d_out=d_att, widths=WidthsEmbedding, dropout=dropout)

        self.Classifier = FeedForward(d_in=d_att, d_out=len_in+1, widths=WidthsEmbedding, dropout=dropout)

    def forward(self, x):
        x = self.Embedding(x)
        x = self.PEIn(x)
        y = self.Latent()
        for Encoder in self.Encoders:
            y = Encoder(y, x)

        y = self.Classifier(y)
        return F.softmax(y, dim=-1)


def ChoseOutput(Pred):
    Arg = Pred.argmax(dim=-1)
    return Arg
