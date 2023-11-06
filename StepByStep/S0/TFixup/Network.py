import torch
import torch.nn as nn
from math import sqrt
from Complete.Transformer.EncoderTransformer import EncoderLayer
from Complete.Transformer.EasyFeedForward import FeedForward
from Complete.Transformer.DecoderTransformer import DecoderLayer
from Complete.TypeClassic.PositionalEncoding import PositionalEncoding

'''
implémentation du réseau de neurones avec une architecture classique de transformer
'''


class Network(nn.Module):

    def __init__(self, d_source, d_target, d_att=32, n_heads=4, network_depth=3,
                 len_source=5, len_target=5, device=torch.device('cpu'), fixup=False):
        super().__init__()
        self.device = device
        self.d_source = d_source
        self.d_target = d_target
        self.d_att = d_att

        self.encoders = nn.ModuleList()
        for i in range(network_depth):
            self.encoders.append(EncoderLayer(d_att=d_att, n_heads=n_heads))

        self.decoders = nn.ModuleList()
        for i in range(network_depth):
            self.decoders.append(DecoderLayer(d_att=d_att, n_heads=n_heads))

        self.source_embedding = FeedForward(d_in=d_source, d_out=d_att, widths=[], dropout=0)
        self.target_embedding = FeedForward(d_in=d_target, d_out=d_att, widths=[], dropout=0)

        self.PE_encoder = PositionalEncoding(d_att=d_att, dropout=0, max_len=len_source, device=device)
        self.PE_decoder = PositionalEncoding(d_att=d_att, dropout=0, max_len=len_target, device=device)

        self.register_buffer("mask_decoder", torch.tril(torch.ones(len_target, len_target)).unsqueeze(0).unsqueeze(0), persistent=False)

        self.output_dembedding = FeedForward(d_in=d_att, d_out=d_target, widths=[], dropout=0)

        self.to(device)

    def forward(self, source, target):
        # source.shape = (batch_size, len_source, d_source)
        # target.shape = (batch_size, len_target, d_target)

        trg = self.PE_decoder(self.target_embedding(target))
        src = self.PE_encoder(self.source_embedding(source))

        # trg.shape = (batch_size, len_target, d_att)
        # src.shape = (batch_size, len_source, d_att)

        for encoder in self.encoders:
            src = encoder(src)

        for decoder in self.decoders:
            trg = decoder(target=trg, source=src, mask=self.mask_decoder)
        # trg.shape = (batch_size, len_target, d_att)

        trg = self.output_dembedding(trg)

        return trg
