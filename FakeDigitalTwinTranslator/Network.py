from Transformer.EasyFeedForward import FeedForward
from Transformer.EncoderTransformer import EncoderLayer
from Perceiver.FlexibleDecoder import DecoderLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class TransformerTranslator(nn.Module):

    def __init__(self, d_source, d_target, d_input_Enc=10, d_input_Dec=10, d_att=10, num_heads=2, num_encoders=3, RPR_len_decoder=32,
                 num_decoders=3, target_len=80, num_flags=4, eps_end=1-1e-1, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.eps_end = eps_end
        self.d_target = d_target
        self.num_flags = num_flags
        self.Encoders = nn.ModuleList()
        for i in range(num_encoders):
            # on se moque de max_len car on n'a ni masque, ni RPR
            self.Encoders.append(EncoderLayer(d_model=d_input_Enc, d_att=d_att, num_heads=num_heads, relative=False))
        self.Encoders[0].relative = True
        self.target_len = target_len
        self.targetStart = nn.Parameter(torch.randn(1, d_target+num_flags))
        self.Decoders = nn.ModuleList()
        for i in range(num_decoders):
            # on doit spécifier target_len car on a un masque dans le décodage
            self.Decoders.append(DecoderLayer(d_model=d_input_Dec, d_att=d_att, num_heads=num_heads, RPR_len=RPR_len_decoder))

        self.SourceEmbedding = FeedForward(d_in=d_source, d_out=d_input_Enc, widths=[16])
        self.TargetEmbedding = FeedForward(d_in=d_target+num_flags, d_out=d_input_Dec, widths=[16])

        self.PhysicalPrediction = FeedForward(d_in=d_input_Dec, d_out=d_target, widths=[16, 32, 16])
        self.FlagsPrediction = FeedForward(d_in=d_input_Dec, d_out=num_flags, widths=[32, 8])

        # Ce vecteur a pour but de déterminer l'action a réaliser, mettre fin à la traduction dans notre cas particulier
        self.ActionPrediction = FeedForward(d_in=d_input_Dec, d_out=1, widths=[32, 8])

        self.to(device)


    def forward(self, source, target, ended=None):
        # target.shape = (batch_size, target_len, d_target+num_flags)
        batch_size, target_len, d_target_raw = target.shape

        # On ajoute le token start au début de target
        trg = torch.cat((self.targetStart.expand(batch_size, 1, d_target_raw), target), 1)
        target_len += 1
        # On pad la trg avec des 0 sur les mots par encore écrits
        if self.target_len-target_len > 0:
            trg = F.pad(trg, (0, 0, 0, self.target_len-target_len))
        else:
            print('target_len :', target_len)
            print('self.target_len :', self.target_len)
            raise ValueError('max len in the target reached')
        trg = self.TargetEmbedding(trg)
        # trg.shape = (batch_size, target_len, d_input_Dec)

        src = self.SourceEmbedding(source)
        for Encoder in self.Encoders:
            src = Encoder(src)
        for Decoder in self.Decoders:
            trg = Decoder(target=trg, source=src)
        y = torch.sum(trg, dim=1)
        Physic, Flags, Action = self.PhysicalPrediction(y), self.FlagsPrediction(y), self.ActionPrediction(y)
        Physic = Physic.unsqueeze(dim=1)
        Flags = Flags.unsqueeze(dim=1)
        Action = Action.unsqueeze(dim=1)

        # Physic.shape = (batch_size, 1, d_target)
        # Flags.shape = (batch_size, 1, num_flags)
        # Action.shape = (batch_size, 1, 1)
        PDW = torch.cat((Physic, Flags), dim=2)

        if ended is not None:
            # ended est une matrice qui va masquer les prédictions des phrases qui sont déjà finies
            # ended.shape = (batch_size, 1, 1)
            PDW = PDW * (1-ended)
            Action = ended
        return PDW, Action

