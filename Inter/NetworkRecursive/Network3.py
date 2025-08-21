import torch
import torch.nn as nn
from Complete.Transformer.EasyFeedForward import FeedForward
from Complete.Transformer.DecoderTransformer import DecoderLayer
from Complete.Transformer.EncoderTransformer import EncoderLayer
from Complete.Transformer.PositionalEncoding import PositionalEncoding
from Complete.Transformer.LearnableModule import LearnableParameters


class TransformerTranslator(nn.Module):

    def __init__(self, d_in, d_out, d_att=32, n_heads=4, n_encoders=3, n_decoders=3, width_FF=[32], widths_embedding=[32],
                 len_in=10, len_out=20, n_mes=6, norm='post', dropout=0, size_tampon_target=12, size_tampon_source=8):
        super().__init__()
        self.n_mes = n_mes
        self.d_in = d_in
        self.d_out = d_out
        self.d_att = d_att
        self.size_tampon_target = size_tampon_target
        self.size_tampon_source = size_tampon_source

        self.enc_embedding = FeedForward(d_in=d_in, d_out=d_att, widths=widths_embedding, dropout=dropout)
        self.dec_embedding = FeedForward(d_in=d_out, d_out=d_att, widths=widths_embedding, dropout=dropout)
        self.mem_embedding = FeedForward(d_in=d_out, d_out=d_att, widths=widths_embedding, dropout=dropout)

        self.enc_pos_encoding = PositionalEncoding(d_att=d_att, dropout=dropout, max_len=len_in + n_mes + 2)
        self.dec_pos_encoding = PositionalEncoding(d_att=d_att, dropout=dropout, max_len=len_out + 1)

        self.end_token = LearnableParameters(torch.normal(0, 1, [1, 1, d_att]))
        self.start_token = LearnableParameters(torch.normal(0, 1, [1, 1, d_att]))
        self.pad_token = LearnableParameters(torch.normal(0, 1, [1, 1, d_att]))

        self.encoders = nn.ModuleList()
        for i in range(n_encoders):
            self.encoders.append(EncoderLayer(d_att=d_att, n_heads=n_heads, norm=norm, width_FF=width_FF, dropout_FF=dropout, dropout_SA=dropout))

        self.decoders = nn.ModuleList()
        for i in range(n_decoders):
            self.decoders.append(DecoderLayer(d_att=d_att, n_heads=n_heads, norm=norm, width_FF=width_FF, dropout_A=dropout, dropout_FF=dropout))

        self.register_buffer("mask_decoder", torch.tril(torch.ones(len_out + 1, len_out + 1)).unsqueeze(0).unsqueeze(0), persistent=False)

        self.mem_decoderFF = FeedForward(d_in=d_att, d_out=d_in + 1, widths=list(widths_embedding.__reversed__()), dropout=0)
        self.pulse_decoder = FeedForward(d_in=d_att, d_out=d_out, widths=list(widths_embedding.__reversed__()), dropout=0)


    def forward(self, source, target, mem_in, input_mask):
        source_pad_mask, source_end_mask, target_pad_mask, target_end_mask, pad_mem_in_mask, pad_mem_out_mask = input_mask

        # source.shape = (batch_size, len_in, d_in)
        # target.shape = (batch_size, len_out, d_out)
        # mem.shape = (batch_size, 2, n_mesureur, d_in + 1)

        trg = self.dec_embedding(target)
        src = self.enc_embedding(source)
        mem_in = self.mem_embedding(mem_in)

        # source.shape = (batch_size, len_in, d_att)
        # target.shape = (batch_size, len_out, d_att)

        src = (src * (1 - source_end_mask) * (1 - source_pad_mask) +
               self.end_token() * source_end_mask +
               self.pad_token() * source_pad_mask)
        trg = (trg * (1 - target_end_mask) * (1 - target_pad_mask) +
               self.end_token() * target_end_mask +
               self.pad_token() * target_pad_mask)
        mem_in = (mem_in * (1 - pad_mem_in_mask) +
                  self.pad_token() * pad_mem_in_mask)

        latent = torch.concat((
            src[:, :self.size_tampon_source],
            self.start_token().expand(trg.size(0), 1, -1),
            src[:, self.size_tampon_source:],
            self.start_token().expand(trg.size(0), 1, -1),
            mem_in
        ), dim=1)

        trg = torch.concat((trg[:, :self.size_tampon_target], self.start_token().expand(trg.size(0), 1, -1), trg[:, self.size_tampon_target:]), dim=1)

        trg = self.dec_pos_encoding(trg)
        latent = self.enc_pos_encoding(latent)

        # trg.shape = (batch_size, len_out + 1, d_att)
        # src.shape = (batch_size, len_in + n_mes + 2, d_att)

        for encoder in self.encoders:
            latent = encoder(latent)

        mem_out = latent[:, -self.n_mes:]
        mem_out = (mem_out - self.pad_token() * pad_mem_out_mask)
        mem_out = self.mem_decoderFF(mem_out)

        for decoder in self.decoders:
            trg = decoder(target=trg, source=latent, mask=self.mask_decoder)
        # trg.shape = (batch_size, len_out + 1, d_att)

        trg[:, :-1] = trg[:, :-1] - self.end_token() * target_end_mask
        trg = self.pulse_decoder(trg)
        # trg.shape = (batch_size, len_out + 1, d_out)

        return trg, mem_out
