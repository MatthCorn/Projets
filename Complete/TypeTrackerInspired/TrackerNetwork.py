import torch
import torch.nn as nn
from Complete.Transformer.EncoderTransformer import EncoderLayer
from Complete.Transformer.EasyFeedForward import FeedForward
from Complete.Transformer.DecoderTransformer import DecoderLayer
from Complete.TypeTrackerInspired.Embedding import TrackerEmbeddingLayer
from Complete.TypeTrackerInspired.PositionalEncoding import PositionalEncoding
from Complete.PreEmbedding import FeaturesAndScaling


'''
implémentation du réseau de neurones avec une architecture inspirée du fonctionnement avec mesureurs du jumeau numérique
'''

def MakeDecoderMask(n_trackers, len_target):
    mask = torch.zeros(n_trackers*len_target, n_trackers*len_target)
    for i in range(len_target):
        mask[i*n_trackers:, i*n_trackers:(i+1)*n_trackers] = 1
    return mask.unsqueeze(0).unsqueeze(0)


class TransformerTranslator(nn.Module):

    def __init__(self, d_pulse, d_pulse_buffed, d_PDW, d_PDW_buffed, d_att=32, n_heads=4, n_encoders=3, n_decoders=3, n_PDWs_memory=10, freq_ech=3,
                 len_source=20, len_target=20, n_flags=4, threshold=7, device=torch.device('cpu'), norm='pre', weights=None, n_trackers=4):
        super().__init__()
        self.device = device
        self.d_pulse = d_pulse
        self.d_pulse_buffed = d_pulse_buffed
        self.d_PDW_buffed = d_PDW_buffed
        self.d_PDW = d_PDW
        self.d_att = d_att
        self.n_trackers = n_trackers
        self.n_flags = n_flags

        self.tokens_encoding = nn.Parameter(torch.randn(3, d_att))
        self.register_buffer("not_token", torch.tensor([1., 0., 0.]).reshape(3, 1).expand(3, d_att), persistent=False)

        self.encoders = nn.ModuleList()
        for i in range(n_encoders):
            self.encoders.append(EncoderLayer(d_att=d_att, n_heads=n_heads, norm=norm))

        self.decoders = nn.ModuleList()
        for i in range(n_decoders):
            self.decoders.append(DecoderLayer(d_att=d_att, n_heads=n_heads, norm=norm))

        self.encoder_decider = EncoderLayer(d_att=d_att, n_heads=n_heads, norm=norm)
        self.resizer_decider = nn.Linear(d_att, 1)
        self.normalizer_decider = nn.Softmax(dim=2)

        self.n_PDWs_memory = n_PDWs_memory

        self.source_pre_embedding = FeaturesAndScaling(threshold, freq_ech, type='source', weights=weights)
        self.target_pre_embedding = FeaturesAndScaling(threshold, freq_ech, type='target', weights=weights)

        self.source_embedding = FeedForward(d_in=d_pulse_buffed, d_out=d_att, widths=[], dropout=0)
        self.target_embedding = FeedForward(d_in=d_PDW_buffed+n_flags, d_out=d_att, widths=[], dropout=0)

        self.tracker_embedding = TrackerEmbeddingLayer(n_trackers=n_trackers, d_att=d_att)

        self.PE_encoder = PositionalEncoding(d_att=d_att, dropout=0, max_len=len_source, device=device)
        self.PE_decoder = PositionalEncoding(d_att=d_att, dropout=0, max_len=len_target, device=device)

        self.register_buffer("mask_decoder", MakeDecoderMask(n_trackers, len_target), persistent=False)

        self.prediction_physics = FeedForward(d_in=d_att, d_out=d_PDW, widths=[16], dropout=0)
        self.prediction_flags = FeedForward(d_in=d_att, d_out=n_flags, widths=[16], dropout=0)


        self.to(device)

    def forward(self, source, target):
        # source.shape = (batch_size, len_source, d_pulse+3)
        # target.shape = (batch_size, len_target, d_PDW+n_flags+3)

        batch_size, len_target, _ = target.shape

        src, type_source = source.split([self.d_pulse, 3], dim=-1)
        trg, type_target = target.split([self.d_PDW + self.n_flags, 3], dim=-1)

        src = self.source_pre_embedding(src)
        trg = self.target_pre_embedding(trg)

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

        trg = self.tracker_embedding(trg)
        # trg.shape = (batch_size, n_trackers*len_target, d_att)

        for encoder in self.encoders:
            src = encoder(src)

        for decoder in self.decoders:
            trg = decoder(target=trg, source=src, mask=self.mask_decoder)
        # trg.shape = (batch_size, n_trackers*len_target, d_att)
        trg = trg.reshape(batch_size*len_target, self.n_trackers, self.d_att)
        # trg.shape = (batch_size*len_target, n_trackers, d_att)

        decision = self.encoder_decider(trg)
        # decision.shape = (batch_size*len_target, n_trackers, d_att)

        trg = trg.reshape(batch_size, len_target, self.n_trackers, self.d_att)
        # trg.shape = (batch_size, len_target, n_trackers, d_att)

        trg = torch.cat((self.prediction_physics(trg), self.prediction_flags(trg)), dim=-1)

        decision = decision.reshape(batch_size, len_target, self.n_trackers, self.d_att)
        # decision.shape = (batch_size, len_target, n_trackers, d_att)
        decision = self.resizer_decider(decision)
        # decision.shape = (batch_size, len_target, n_trackers, 1)
        decision = self.normalizer_decider(decision)

        trg = self.HookMatmul(trg.transpose(-1, -2), decision).squeeze(-1)
        # trg.shape = (batch_size, len_target, d_PDW + n_flags)

        return trg

    def HookMatmul(self, x, y):
        # custom matmul pour pouvoir récupérer entrées et sorties avec un hook
        return torch.matmul(x, y)