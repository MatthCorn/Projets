from Transformer.EasyFeedForward import FeedForward
from Transformer.EncoderTransformer import EncoderLayer
from Transformer.DecoderTransformer import DecoderLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerTranslator(nn.Module):

    def __init__(self, d_source, d_target, d_input_Enc=10, d_input_Dec=10, d_att=10, num_heads=2, num_encoders=3, num_decoders=3, target_len=80, num_flags=4):
        super().__init__()
        self.d_target = d_target
        self.Encoders = nn.ModuleList()
        for i in range(num_encoders):
            # on se moque de max_len car on n'a ni masque, ni RPR
            self.Encoders.append(EncoderLayer(d_model=d_input_Enc, d_att=d_att, num_heads=num_heads, relative=False))
        self.Encoders[0].relative = True
        self.target_len = target_len
        self.targetStart = nn.Parameter(torch.randn(1, d_target))
        self.Decoders = nn.ModuleList()
        for i in range(num_decoders):
            # on doit spécifier target_len car on a un masque dans le décodage
            self.Decoders.append(DecoderLayer(d_model=d_input_Dec, d_att=d_att, num_heads=num_heads, target_len=target_len))

        self.SourceEmbedding = FeedForward(d_in=d_source, d_out=d_input_Enc, widths=[16])
        self.TargetEmbedding = FeedForward(d_in=d_target, d_out=d_input_Dec, widths=[16])

        self.PhysicalPrediction = FeedForward(d_in=d_input_Dec, d_out=d_target, widths=[16, 32, 16])
        self.FlagsPrediction = FeedForward(d_in=d_input_Dec, d_out=num_flags+1, widths=[32, 8])

    def forward(self, source, target):
        src = self.SourceEmbedding(source)
        # target.shape = (batch_size, target_len, d_target)
        batch_size, _, _ = target.shape
        # On ajoute le token start au début de target
        trg = torch.cat((self.targetStart.expand(batch_size, 1, self.d_target), target), 1)
        # On pad la trg avec des 0 sur les mots par encore écrits
        if self.target_len-len(target) > 0:
            trg = F.pad(trg, (0, 0, 0, self.target_len-len(target)))
        else:
            raise ValueError('max len in the target reached')
        trg = self.TargetEmbedding(trg)
        # trg.shape = (batch_size, target_len, d_input_Dec)
        for Encoder in self.Encoders:
            src = Encoder(src)
        for Decoder in self.Decoders:
            trg = Decoder(target=trg, source=src)
        y = torch.mean(trg, dim=1)
        return self.PhysicalPrediction(y), F.softmax(self.FlagsPrediction(y))
