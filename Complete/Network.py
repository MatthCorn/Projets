import math
import torch
import torch.nn as nn
from Transformer.EncoderTransformer import EncoderLayer
from Transformer.EasyFeedForward import FeedForward
from Transformer.DecoderTransformer import DecoderLayer
from FDTTranslator.Bursts.TrackerInspired.Embedding import TrackerEmbeddingLayer
def MakePermutMask(n_tracker, target_len):
    Permut = (torch.tril(torch.ones(n_tracker * target_len, n_tracker * target_len), n_tracker) -
              torch.tril(torch.ones(n_tracker * target_len, n_tracker * target_len), n_tracker - 1) +
              torch.tril(torch.ones(n_tracker * target_len, n_tracker * target_len), -n_tracker * (target_len - 1)) -
              torch.tril(torch.ones(n_tracker * target_len, n_tracker * target_len), -n_tracker * (target_len - 1) - 1))
    Powers = tuple(torch.linalg.matrix_power(Permut, i) for i in range(target_len))
    return torch.tril(sum(Powers)).unsqueeze(0).unsqueeze(0)

def MakeTrackerMask(n_tracker, target_len):
    mask = torch.zeros(n_tracker*target_len, n_tracker*target_len)
    for i in range(target_len):
        mask[i*n_tracker:, i*n_tracker:(i+1)*n_tracker] = 1
    return mask.unsqueeze(0).unsqueeze(0)


class TransformerTranslator(nn.Module):

    def __init__(self, d_pulse, d_PDW, d_latent=32, num_heads=4, num_encoders=3, n_tracker=4,
                 num_decoders=3, NbPDWsMemory=10, target_len=20, num_flags=4, device=torch.device('cpu'), FPIC=False):
        super().__init__()
        self.device = device
        self.d_PDW = d_PDW
        self.d_latent = d_latent
        self.n_tracker = n_tracker
        self.num_flags = num_flags

        self.Encoders = nn.ModuleList()
        for i in range(num_encoders):
            self.Encoders.append(EncoderLayer(d_model=d_latent, d_att=d_latent, num_heads=num_heads, relative=False, MHADropout=0, FFDropout=0))

        self.Decoders = nn.ModuleList()
        Decoder = DecoderLayer(d_model=d_latent, d_att=d_latent, num_heads=num_heads, target_len=target_len*n_tracker, relative=False, MHADropout=0, FFDropout=0)
        Decoder.SelfAttentionLayer.mask = MakePermutMask(n_tracker, target_len)
        self.Decoders.append(Decoder)
        for i in range(num_decoders-1):
            Decoder = DecoderLayer(d_model=d_latent, d_att=d_latent, num_heads=num_heads, target_len=target_len*n_tracker, relative=False, MHADropout=0, FFDropout=0)
            Decoder.SelfAttentionLayer.mask = MakeTrackerMask(n_tracker, target_len)
            self.Decoders.append(Decoder)

        self.NbPDWsMemory = NbPDWsMemory
        self.FPIC = FPIC



        self.SourceEmbedding = FeedForward(d_in=d_pulse, d_out=d_latent, widths=[], dropout=0)
        self.TargetEmbedding = FeedForward(d_in=d_PDW+num_flags, d_out=d_latent, widths=[], dropout=0)
        self.TrackerEmbedding = TrackerEmbeddingLayer(n_tracker=n_tracker, d_input_decoder=d_latent, d_att=d_latent)

        self.PhysicalPrediction = FeedForward(d_in=d_latent, d_out=d_PDW, widths=[16], dropout=0)
        self.FlagsPrediction = FeedForward(d_in=d_latent, d_out=num_flags, widths=[8], dropout=0)

        # Ce vecteur a pour but de déterminer l'action a réaliser, mettre fin à la traduction dans notre cas particulier
        self.ActionPrediction = FeedForward(d_in=d_latent, d_out=1, widths=[8], dropout=0)

        if self.FPIC:
            # Il faut donner à ce classifieur l'impulsion fictive transportée dans l'espace des PDWs
            self.PulseToPDWs = FeedForward(d_in=d_latent, d_out=d_latent, widths=[], dropout=0)

        self.PEEncoder = PositionalEncoding(d_model=d_latent, dropout=0, max_len=target_len, device=device)
        self.PEDecoder = PositionalEncoding(d_model=d_latent, dropout=0, max_len=target_len, device=device)

        self.to(device)

    def forward(self, source, target):
        # target.shape = (batch_size, target_len, d_target+num_flags)
        batch_size, len_source, _ = source.shape

        trg = self.PEDecoder(self.TargetEmbedding(target))
        # trg.shape = (batch_size, target_len, d_latent)
        trg = self.TrackerEmbedding(trg)
        # trg.shape = (batch_size, n_tracker*target_len, d_latent)

        src = self.PEEncoder(self.SourceEmbedding(source))
        # src.shape = (batch_size, len_source, d_latent)

        if self.FPIC:
            src, FakePulse = torch.split(src, len_source - 1, dim=1)

        for Encoder in self.Encoders:
            src = Encoder(src)

        for Decoder in self.Decoders:
            trg = Decoder(target=trg, source=src)
        # trg.shape = (batch_size, n_tracker*target_len, d_latent)

        trg = trg.reshape(batch_size, -1, self.n_tracker*self.d_latent)[:, :, :self.d_latent]
        # trg.shape = (batch_size, target_len, d_latent)

        if self.FPIC:
            Physic, Flags, Action = self.PhysicalPrediction(trg), self.FlagsPrediction(trg), self.ActionPrediction(trg - self.PulseToPDWs(FakePulse))
        else:
            Physic, Flags, Action = self.PhysicalPrediction(trg), self.FlagsPrediction(trg), self.ActionPrediction(trg)

        # Physic.shape = (batch_size, target_len, d_PDW)
        # Flags.shape = (batch_size, target_len, num_flags)
        # Action.shape = (batch_size, target_len, 1)
        PDW = torch.cat((Physic, Flags), dim=2)
        return PDW, Action


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0., max_len: int = 5000, device=torch.device('cpu')):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.to(device=device))

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)