from Transformer.EasyFeedForward import FeedForward
from Transformer.EncoderTransformer import EncoderLayer
from Perceiver.FlexibleDecoder import DecoderLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

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


    def forward(self, source, target):
        # target.shape = (batch_size, target_len, d_target+num_flags)
        batch_size, target_len, d_target_raw = target.shape

        # On ajoute le token start au début de target
        trg = torch.cat((self.targetStart.expand(batch_size, 1, d_target_raw), target), 1)
        # On pad la trg avec des 0 sur les mots par encore écrits
        if self.target_len-target_len >= 0:
            trg = F.pad(trg, (0, 0, 0, self.target_len-target_len))
        else:
            print('target_len :', target_len)
            print('self.target_len :', self.target_len)
            raise ValueError('max len in the target reached')
        trg = self.TargetEmbedding(trg)
        # trg.shape = (batch_size, self.target_len, d_input_Dec)

        src = self.SourceEmbedding(source)
        for Encoder in self.Encoders:
            src = Encoder(src)
        for Decoder in self.Decoders:
            trg = Decoder(target=trg, source=src)
        # trg.shape = (batch_size, target_len, d_input_Dec)
        Physic, Flags, Action = self.PhysicalPrediction(trg), self.FlagsPrediction(trg), self.ActionPrediction(trg)

        # Physic.shape = (batch_size, target_len, d_target)
        # Flags.shape = (batch_size, target_len, num_flags)
        # Action.shape = (batch_size, target_len, 1)
        PDW = torch.cat((Physic, Flags), dim=2)
        return PDW, Action

    def translate(self, source):
        batch_size = len(source)

        Translation = torch.zeros(size=(batch_size, self.len_target, self.d_target + self.num_flags), device=self.device)
        # Translation.shape = (batch_size, len_target, d_target + num_flags)

        with torch.no_grad():
            for i in range(self.len_target):
                if i == self.len_target - 1:
                    Translation, Actions = self.forward(source=source, target=Translation)
                else:
                    Translation, _ = self.forward(source=source, target=Translation)
                Translation = Translation[:, 1:]
                Translation[:, self.len_target + 1:] = 0

            Actions = Actions[:, 1:, 0]
            MaskEndTranslation = torch.zeros(size=(batch_size, self.len_target, self.d_target + self.num_flags), device=self.device)
            for i in range(batch_size):
                for j in range(self.len_target):
                    if Actions[i, j] > 0.5:
                        MaskEndTranslation[i, j] = 1
                    else:
                        break


        return Translation, MaskEndTranslation