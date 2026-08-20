import torch
import torch.nn as nn
from Complete.Transformer.EncoderTransformer import EncoderLayer
from Complete.Transformer.EasyFeedForward import FeedForward
from Complete.Transformer.DecoderTransformer import DecoderLayer
from Complete.Transformer.PositionalEncoding import PositionalEncoding
from Complete.Transformer.LearnableModule import LearnableParameters


class TransformerTranslator(nn.Module):

    def __init__(self, d_in, d_out, d_att=32, n_heads=4, n_encoders=3, n_decoders=3, width_FF=[32], widths_embedding=[32],
                 len_in=10, len_out=20, norm='post', dropout=0):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_att = d_att
        self.n_heads = n_heads
        self.d_head = d_att // n_heads

        self.enc_embedding = FeedForward(d_in=d_in, d_out=d_att, widths=widths_embedding, dropout=dropout)
        self.dec_embedding = FeedForward(d_in=d_out, d_out=d_att, widths=widths_embedding, dropout=dropout)

        self.enc_pos_encoding = PositionalEncoding(d_att=d_att, dropout=dropout, max_len=len_in)
        self.dec_pos_encoding = PositionalEncoding(d_att=d_att, dropout=dropout, max_len=len_out + 1)

        self.end_token = LearnableParameters(torch.normal(0, 1, [1, 1, d_att]))
        self.start_token = LearnableParameters(torch.normal(0, 1, [1, 1, d_att]))


        self.encoders = nn.ModuleList()
        for i in range(n_encoders):
            self.encoders.append(EncoderLayer(d_att=d_att, n_heads=n_heads, norm=norm, width_FF=width_FF, dropout_FF=dropout, dropout_SA=dropout))

        self.decoders = nn.ModuleList()
        for i in range(n_decoders):
            self.decoders.append(DecoderLayer(d_att=d_att, n_heads=n_heads, norm=norm, width_FF=width_FF, dropout_A=dropout, dropout_FF=dropout))

        self.register_buffer("mask_decoder", torch.tril(torch.ones(len_out + 1, len_out + 1)).unsqueeze(0).unsqueeze(0), persistent=False)

        self.last_decoder = FeedForward(d_in=d_att, d_out=d_out, widths=[16], dropout=0)
        self.past_kv = None


    def forward(self, source, target, target_mask=None):
        # source.shape = (batch_size, len_in, d_in)
        # target.shape = (batch_size, len_out, d_out)

        trg = self.dec_embedding(target)
        src = self.enc_embedding(source)

        # source.shape = (batch_size, len_in, d_att)
        # target.shape = (batch_size, len_out, d_att)

        trg = torch.concat((self.start_token().expand(trg.size(0), 1, -1), trg), dim=1)

        trg = self.dec_pos_encoding(trg)
        src = self.enc_pos_encoding(src)
        # trg.shape = (batch_size, len_out + 1, d_att)
        # src.shape = (batch_size, len_in, d_att)

        for encoder in self.encoders:
            src = encoder(src)

        for decoder in self.decoders:
            trg = decoder(target=trg, source=src, mask=self.mask_decoder)
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

    def recursive_eval(self, source, target, n=0):

        trg = self.dec_embedding(target)
        src = self.enc_embedding(source)

        trg = torch.concat((self.start_token().expand(trg.size(0), 1, -1), trg), dim=1)

        trg = self.dec_pos_encoding(trg)
        src = self.enc_pos_encoding(src)

        for encoder in self.encoders:
            src = encoder(src)

        for decoder in self.decoders:
            trg = decoder(target=trg, source=src, mask=self.mask_decoder)

        add_mask = torch.zeros((1, target.shape[1] + 1, 1), device=target.device)
        add_mask[0, n] += 1

        is_end = torch.norm(self.last_decoder(trg - self.end_token()) * add_mask)

        trg = self.last_decoder(trg)

        return trg, is_end

    def step(self, source, target, offset=0):
        """
        Mode Inférence Pas-à-Pas (Bufferisé).
        buffer: contient l'historique brut des embeddings [Batch, Hidden, T_history]
        """
        # 1. Gestion du Buffer
        if self.past_kv is None:
            batch_size, *_ = source.shape
            self.past_kv = [[[],
                             [torch.zeros(batch_size, 0, self.n_heads, self.d_head, device=source.device),
                             torch.zeros(batch_size, self.n_heads, 0, self.d_head, device=source.device)].copy()]
                            for _ in range(len(self.decoders))]

        # 2. Calcul de la sortie de l'encodeur
        src = self.enc_embedding(source)
        src = self.enc_pos_encoding(src)
        for encoder in self.encoders:
            src = encoder(src)

        # 3. Embeddings de l'instant t
        if target.shape[1] == 0:
            trg = self.start_token().expand(target.size(0), 1, -1)
        else:
            trg = self.dec_embedding(target)
            trg = self.dec_pos_encoding(trg, offset=offset)
        # trg.shape = (batch_size, 1, d_att)

        for i, decoder in enumerate(self.decoders):
            trg = decoder(target=trg, source=src, mask=self.mask_decoder, past_kv=self.past_kv[i])

        trg = self.last_decoder(trg)
        return trg

    def reset_cache(self):
        self.past_kv = None

if __name__ == '__main__':
    N = TransformerTranslator(10, 11, len_in=37, len_out=41)
    X = torch.normal(0, 1, (1000, 37, 10))
    Y = torch.normal(0, 1, (1000, 41, 11))

    Z1 = N(X, Y)[:, 1:]

    y_list = []
    y = torch.zeros(1000, 0, 11)
    for k in range(Y.shape[1]):
        y = N.step(X, y, offset=k)
        y_list.append(y)
        y = Y[:, k:k+1]
    Z2 = torch.cat(y_list, dim=1)

    print('Résultats faux mais temps  de calcul réaliste')
