import torch
import torch.nn as nn
from Complete.Transformer.EncoderTransformer import EncoderLayer
from Complete.Transformer.EasyFeedForward import FeedForward
from Complete.Transformer.DecoderTransformer import DecoderLayer
from Complete.Embedding import TrackerEmbeddingLayer
from Complete.PositionalEncoding import ClassicPositionalEncoding

def MakePermutMask(n_tracker, len_target):
    Permut = (torch.tril(torch.ones(n_tracker * len_target, n_tracker * len_target), n_tracker) -
              torch.tril(torch.ones(n_tracker * len_target, n_tracker * len_target), n_tracker - 1) +
              torch.tril(torch.ones(n_tracker * len_target, n_tracker * len_target), -n_tracker * (len_target - 1)) -
              torch.tril(torch.ones(n_tracker * len_target, n_tracker * len_target), -n_tracker * (len_target - 1) - 1))
    Powers = tuple(torch.linalg.matrix_power(Permut, i) for i in range(len_target))
    return torch.tril(sum(Powers)).unsqueeze(0).unsqueeze(0)

def MakeTrackerMask(n_tracker, len_target):
    mask = torch.zeros(n_tracker*len_target, n_tracker*len_target)
    for i in range(len_target):
        mask[i*n_tracker:, i*n_tracker:(i+1)*n_tracker] = 1
    return mask.unsqueeze(0).unsqueeze(0)


class TransformerTranslator(nn.Module):

    def __init__(self, d_pulse, d_PDW, d_att=32, n_heads=4, n_encoders=3, n_tracker=4,
                 n_decoders=3, n_PDWs_memory=10, len_target=20, n_flags=4, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.d_PDW = d_PDW
        self.d_att = d_att
        self.n_tracker = n_tracker
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
        self.tracker_embedding = TrackerEmbeddingLayer(n_tracker=n_tracker, d_att=d_att)

        self.physical_prediction = FeedForward(d_in=d_att, d_out=d_PDW, widths=[16], dropout=0)
        self.flags_prediction = FeedForward(d_in=d_att, d_out=n_flags, widths=[8], dropout=0)
        self.action_prediction = FeedForward(d_in=d_att, d_out=1, widths=[8], dropout=0)

        self.PE_encoder = ClassicPositionalEncoding(d_model=d_att, dropout=0, max_len=len_target, device=device)
        self.PE_decoder = ClassicPositionalEncoding(d_model=d_att, dropout=0, max_len=len_target, device=device)

        self.to(device)

    def forward(self, source, target):
        # target.shape = (batch_size, len_target, d_target+num_flags)
        batch_size, len_source, _ = source.shape

        trg = self.PE_decoder(self.target_embedding(target))
        # trg.shape = (batch_size, len_target, d_att)
        trg = self.tracker_embedding(trg)
        # trg.shape = (batch_size, n_tracker*len_target, d_att)

        src = self.PE_encoder(self.source_embedding(source))
        # src.shape = (batch_size, len_source, d_att)

        for Encoder in self.Encoders:
            src = Encoder(src)

        for Decoder in self.Decoders:
            trg = Decoder(target=trg, source=src)
        # trg.shape = (batch_size, n_tracker*len_target, d_att)

        trg = trg.reshape(batch_size, -1, self.n_tracker*self.d_att)[:, :, :self.d_att]
        # trg.shape = (batch_size, len_target, d_att)


        physic, flags, action = self.physical_prediction(trg), self.flags_prediction(trg), self.action_prediction(trg)

        # Physic.shape = (batch_size, len_target, d_PDW)
        # Flags.shape = (batch_size, len_target, n_flags)
        # Action.shape = (batch_size, len_target, 1)
        PDW = torch.cat((physic, flags), dim=2)
        return PDW, action
