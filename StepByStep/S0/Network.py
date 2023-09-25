from Transformer.EasyFeedForward import FeedForward
from Transformer.EncoderTransformer import EncoderLayer
from Perceiver.FlexibleDecoder import DecoderLayer
import torch
import torch.nn as nn

class TransformerTranslator(nn.Module):

    def __init__(self, d_source, d_target, d_input_Enc=10, d_input_Dec=10, d_att=10, num_heads=2, num_encoders=3, RPR_len_decoder=32,
                 num_decoders=3, NbPDWsMemory=10, target_len=80, num_flags=4, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.d_target = d_target
        self.num_flags = num_flags
        self.Encoders = nn.ModuleList()
        for i in range(num_encoders):
            # si relative=True, on s'assure juste que max_len=64 est plus grand que le nombre d'impulsions fournies à l'IA
            self.Encoders.append(EncoderLayer(d_model=d_input_Enc, d_att=d_att, num_heads=num_heads, relative=False, MHADropout=0, FFDropout=0))
        self.Encoders[0].relative = True
        self.target_len = target_len
        self.NbPDWsMemory = NbPDWsMemory
        self.Decoders = nn.ModuleList()
        for i in range(num_decoders):
            # on doit spécifier target_len car on a un masque dans le décodage
            self.Decoders.append(DecoderLayer(d_model=d_input_Dec, d_att=d_att, num_heads=num_heads, RPR_len=RPR_len_decoder, MHADropout=0, FFDropout=0))

        self.SourceEmbedding = FeedForward(d_in=d_source, d_out=d_input_Enc, widths=[16], dropout=0)
        self.TargetEmbedding = FeedForward(d_in=d_target+num_flags, d_out=d_input_Dec, widths=[16], dropout=0)

        self.PhysicalPrediction = FeedForward(d_in=d_input_Dec, d_out=d_target, widths=[16, 32, 16], dropout=0)

        self.to(device)


    def forward(self, source, target):
        trg = self.TargetEmbedding(target)
        # trg.shape = (batch_size, self.target_len, d_input_Dec)

        src = self.SourceEmbedding(source)

        for Encoder in self.Encoders:
            src = Encoder(src)
        for Decoder in self.Decoders:
            trg = Decoder(target=trg, source=src)
        # trg.shape = (batch_size, target_len, d_input_Dec)

        Physic = self.PhysicalPrediction(trg)

        # Physic.shape = (batch_size, target_len, d_target)
        return Physic