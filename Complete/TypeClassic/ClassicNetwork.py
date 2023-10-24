import torch
import torch.nn as nn
from Complete.Transformer.EncoderTransformer import EncoderLayer
from Complete.Transformer.EasyFeedForward import FeedForward
from Complete.Transformer.DecoderTransformer import DecoderLayer
from Complete.TypeClassic.PositionalEncoding import PositionalEncoding

'''
implémentation du réseau de neurones avec une architecture classique de transformer
'''


class TransformerTranslator(nn.Module):

    def __init__(self, d_pulse, d_PDW, d_att=32, n_heads=4, n_encoders=3, n_decoders=3,
                 n_PDWs_memory=10, len_target=20, n_flags=4, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.d_PDW = d_PDW
        self.d_att = d_att
        self.n_flags = n_flags

        self.encoders = nn.ModuleList()
        for i in range(n_encoders):
            self.encoders.append(EncoderLayer(d_att=d_att, n_heads=n_heads))

        self.decoders = nn.ModuleList()
        for i in range(n_decoders):
            self.decoders.append(DecoderLayer(d_att=d_att, n_heads=n_heads))

        self.n_PDWs_memory = n_PDWs_memory

        self.source_embedding = FeedForward(d_in=d_pulse, d_out=d_att, widths=[], dropout=0)
        self.target_embedding = FeedForward(d_in=d_PDW+n_flags, d_out=d_att, widths=[], dropout=0)

        self.PE_encoder = PositionalEncoding(d_att=d_att, dropout=0, max_len=len_target, device=device)
        self.PE_decoder = PositionalEncoding(d_att=d_att, dropout=0, max_len=len_target, device=device)

        self.register_buffer("mask_decoder", torch.tril(torch.ones(len_target, len_target)).unsqueeze(0).unsqueeze(0), persistent=False)

        self.prediction_physics = FeedForward(d_in=d_att, d_out=d_PDW, widths=[16], dropout=0)
        self.prediction_flags = FeedForward(d_in=d_att, d_out=n_flags, widths=[16], dropout=0)

        # Ce vecteur a pour but de déterminer l'action a réaliser, mettre fin à la traduction dans notre cas particulier
        self.prediction_action = FeedForward(d_in=d_att, d_out=1, widths=[32, 8], dropout=0)

        self.to(device)

    def forward(self, source, target):
        # target.shape = (batch_size, len_target, d_target+num_flags)
        batch_size, len_target, _ = target.shape

        trg = self.PE_decoder(self.target_embedding(target))
        # trg.shape = (batch_size, len_target, d_att)

        src = self.PE_encoder(self.source_embedding(source))
        # src.shape = (batch_size, len_source, d_att)

        for encoder in self.encoders:
            src = encoder(src)

        for decoder in self.decoders:
            trg = decoder(target=trg, source=src, mask=self.mask_decoder)
        # trg.shape = (batch_size, len_target, d_att)

        trg = torch.cat((self.prediction_physics(trg), self.prediction_flags(trg), self.prediction_action(trg)), dim=2)

        PDW, action = trg.split([self.d_PDW + self.n_flags, 1], dim=-1)

        return PDW, action
