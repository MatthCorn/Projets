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

    def __init__(self, d_pulse, d_PDW, d_att=32, n_heads=4, n_encoders=3, n_decoders=3, n_PDWs_memory=10,
                 len_source=10, len_target=20, n_flags=4, device=torch.device('cpu'), pre_norm=False):
        super().__init__()
        self.device = device
        self.d_pulse = d_pulse
        self.d_PDW = d_PDW
        self.d_att = d_att
        self.n_flags = n_flags

        self.tokens_encoding = nn.Parameter(torch.randn(3, d_att))
        self.register_buffer("not_token", torch.tensor([1., 0., 0.]).reshape(3, 1).expand(3, d_att), persistent=False)

        self.encoders = nn.ModuleList()
        for i in range(n_encoders):
            self.encoders.append(EncoderLayer(d_att=d_att, n_heads=n_heads, pre_norm=pre_norm))

        self.decoders = nn.ModuleList()
        for i in range(n_decoders):
            self.decoders.append(DecoderLayer(d_att=d_att, n_heads=n_heads, pre_norm=pre_norm))

        self.n_PDWs_memory = n_PDWs_memory

        self.source_embedding = FeedForward(d_in=d_pulse, d_out=d_att, widths=[], dropout=0)
        self.target_embedding = FeedForward(d_in=d_PDW+n_flags, d_out=d_att, widths=[], dropout=0)

        self.PE_encoder = PositionalEncoding(d_att=d_att, dropout=0, max_len=len_source, device=device)
        self.PE_decoder = PositionalEncoding(d_att=d_att, dropout=0, max_len=len_target, device=device)

        self.register_buffer("mask_decoder", torch.tril(torch.ones(len_target, len_target)).unsqueeze(0).unsqueeze(0), persistent=False)

        self.prediction_physics = FeedForward(d_in=d_att, d_out=d_PDW, widths=[16], dropout=0)
        self.prediction_flags = FeedForward(d_in=d_att, d_out=n_flags, widths=[16], dropout=0)


        self.to(device)

    def forward(self, source, target):
        # source.shape = (batch_size, len_source, d_pulse+3)
        # target.shape = (batch_size, len_target, d_PDW+n_flags+3)

        batch_size, len_target, _ = target.shape

        src, type_source = source.split([self.d_pulse, 3], dim=-1)
        trg, type_target = target.split([self.d_PDW + self.n_flags, 3], dim=-1)

        trg = self.target_embedding(trg)
        src = self.source_embedding(src)

        # On fait les opérations suivantes en une seule ligne pour ne pas garder en mémoire des tenseurs qui ne sont plus utiles
        # token_target = torch.matmul(type_target, self.tokens_encoding)
        # not_token_mask_target = torch.matmul(type_target, self.not_token)
        #
        # token_source = torch.matmul(type_source, self.tokens_encoding)
        # not_token_mask_source = torch.matmul(type_source, self.not_token)
        #
        # trg = not_token_mask_target*trg + (1-not_token_mask_target)*token_target
        #
        # src = not_token_mask_source*src + (1-not_token_mask_source)*token_source

        not_token_mask_target = torch.matmul(type_target, self.not_token)
        not_token_mask_source = torch.matmul(type_source, self.not_token)
        trg = not_token_mask_target * trg + (1 - not_token_mask_target) * torch.matmul(type_target, self.tokens_encoding)
        src = not_token_mask_source * src + (1 - not_token_mask_source) * torch.matmul(type_source, self.tokens_encoding)

        trg = self.PE_decoder(trg)
        src = self.PE_encoder(src)

        # trg.shape = (batch_size, len_target, d_att)
        # src.shape = (batch_size, len_source, d_att)

        for encoder in self.encoders:
            src = encoder(src)

        for decoder in self.decoders:
            trg = decoder(target=trg, source=src, mask=self.mask_decoder)
        # trg.shape = (batch_size, len_target, d_att)

        trg = torch.cat((self.prediction_physics(trg), self.prediction_flags(trg)), dim=2)

        return trg
