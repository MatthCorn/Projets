import torch
import torch.nn as nn
from PointTransformer.EncoderTransformer import EncoderLayer
from Complete.Transformer.EasyFeedForward import FeedForward
from PointTransformer.DecoderTransformer import DecoderLayer
from PointTransformer.PositionalEncoding import PositionalEncoding
from Complete.Transformer.LearnableModule import LearnableParameters


class PointTransformer(nn.Module):

    def __init__(self, d_in, d_out, d_att=32, d_group=1, n_encoders=3, n_decoders=3, widths_embedding=[32],
                 len_in=10, len_out=20, norm='post', dropout=0):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_att = d_att
        self.d_group = d_group
        n_group, remainder = divmod(d_att, d_group)
        if remainder:
            raise ValueError("incompatible `d_att` and `d_group`")
        self.n_group = n_group
        self.len_in = len_in
        self.len_out = len_out

        self.enc_embedding = FeedForward(d_in=d_in, d_out=d_att, widths=widths_embedding, dropout=dropout)
        self.dec_embedding = FeedForward(d_in=d_out, d_out=d_att, widths=widths_embedding, dropout=dropout)

        self.end_token = LearnableParameters(torch.normal(0, 1, [1, 1, d_att]))
        self.start_token = LearnableParameters(torch.normal(0, 1, [1, 1, d_in]))

        self.positional_encoding = PositionalEncoding(d_pos=3, n_group=n_group)

        self.encoders = nn.ModuleList()
        for i in range(n_encoders):
            self.encoders.append(EncoderLayer(d_att=d_att, d_group=d_group, norm=norm, dropout_FF=dropout, dropout_SA=dropout))
        self.decoders = nn.ModuleList()
        for i in range(n_decoders):
            self.decoders.append(DecoderLayer(d_att=d_att, d_group=d_group, norm=norm, dropout_A=dropout, dropout_FF=dropout))

        self.register_buffer("mask_decoder", torch.tril(torch.ones(len_out + 1, len_out + 1)).unsqueeze(-1), persistent=False)

        self.last_decoder = FeedForward(d_in=d_att, d_out=d_out, widths=[16], dropout=0)


    def forward(self, source, target, target_mask=None):
        # source.shape = (batch_size, len_in, d_in)
        trg = torch.concat((self.start_token().expand(target.size(0), 1, -1), target), dim=1)
        # target.shape = (batch_size, len_out + 1, d_out)


        ts_pos_diff = trg[:, :, [0, 2, 3]].unsqueeze(-2) - source[:, :, [0, 2, 3]].unsqueeze(-3)
        # ts_pos_diff.shape = (batch_size, len_out, len_in, d_pos = 3)
        positional_adding_bias_ts = self.positional_encoding.add(ts_pos_diff)
        # positional_adding_bias_ts.shape = (batch_size, len_out, len_in, n_group)
        positional_multiplying_bias_ts = self.positional_encoding.mult(ts_pos_diff)
        # positional_multiplying_bias_ts.shape = (batch_size, len_out, len_in, n_group)

        tt_pos_diff = trg[:, :, [0, 2, 3]].unsqueeze(-2) - trg[:, :, [0, 2, 3]].unsqueeze(-3)
        # tt_pos_diff.shape = (batch_size, len_seq, len_seq, d_pos = 3)
        positional_adding_bias_tt = self.positional_encoding.add(tt_pos_diff)
        # positional_adding_bias_tt.shape = (batch_size, len_seq, len_seq, n_group)
        positional_multiplying_bias_tt = self.positional_encoding.mult(tt_pos_diff)
        # positional_multiplying_bias_tt.shape = (batch_size, len_seq, len_seq, n_group)


        ss_pos_diff = source[:, :, [0, 2, 3]].unsqueeze(-2) - source[:, :, [0, 2, 3]].unsqueeze(-3)
        positional_adding_bias_ss = self.positional_encoding.add(ss_pos_diff)
        positional_multiplying_bias_ss = self.positional_encoding.mult(ss_pos_diff)

        trg = self.dec_embedding(trg)
        src = self.enc_embedding(source)

        # source.shape = (batch_size, len_in, d_att)
        # target.shape = (batch_size, len_out, d_att)


        # trg.shape = (batch_size, len_out + 1, d_att)

        for encoder in self.encoders:
            src = encoder(src, positional_adding_bias_ss, positional_multiplying_bias_ss)

        for decoder in self.decoders:
            trg = decoder(trg, src, positional_adding_bias_tt, positional_multiplying_bias_tt,
                          positional_adding_bias_ts, positional_multiplying_bias_ts, mask=self.mask_decoder)
        # trg.shape = (batch_size, len_out + 1, d_att)

        if target_mask is not None:
            add_mask, mult_mask = target_mask

            trg = trg - self.end_token() * add_mask
            trg = self.last_decoder(trg)
            # trg.shape = (batch_size, len_out + 1, d_out)

            trg = trg * mult_mask

        else:
            trg = self.last_decoder(trg)

        return trg

    def rec_forward(self, source):
        i = 0
        device = self.start_token().device
        target = torch.zeros((1, self.len_out, self.d_out), device=device)
        while i < self.len_out:
            target = self.forward(source, target)[:, :-1, :]
            i += 1
        return target.detach(), torch.norm(target - self.last_decoder(self.end_token()), dim=-1)