from Transformer.EasyFeedForward import FeedForward
from Transformer.EncoderTransformer import EncoderLayer
from Perceiver.FlexibleDecoder import DecoderLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerTranslator(nn.Module):

    def __init__(self, d_source, d_target, d_input_Enc=10, d_input_Dec=10, d_att=10, num_heads=2, num_encoders=3, RPR_len_decoder=32,
                 num_decoders=3, NbPDWsMemory=10, target_len=80, num_flags=4, eps_end=1-1e-1, device=torch.device('cpu'), FPIC=False):
        super().__init__()
        self.device = device
        self.eps_end = eps_end
        self.d_target = d_target
        self.num_flags = num_flags
        self.Encoders = nn.ModuleList()
        for i in range(num_encoders):
            # si relative=True, on s'assure juste que max_len=64 est plus grand que le nombre d'impulsions fournies à l'IA
            self.Encoders.append(EncoderLayer(d_model=d_input_Enc, d_att=d_att, num_heads=num_heads, relative=True, MHADropout=0, FFDropout=0))
        self.target_len = target_len
        self.NbPDWsMemory = NbPDWsMemory
        self.FPIC = FPIC
        self.targetStart = nn.Parameter(torch.randn(1, d_target+num_flags))
        self.Decoders = nn.ModuleList()
        for i in range(num_decoders):
            # on doit spécifier target_len car on a un masque dans le décodage
            self.Decoders.append(DecoderLayer(d_model=d_input_Dec, d_att=d_att, num_heads=num_heads, RPR_len=RPR_len_decoder, MHADropout=0, FFDropout=0))

        self.SourceEmbedding = FeedForward(d_in=d_source, d_out=d_input_Enc, widths=[16], dropout=0)
        self.TargetEmbedding = FeedForward(d_in=d_target+num_flags, d_out=d_input_Dec, widths=[16], dropout=0)

        self.PhysicalPrediction = FeedForward(d_in=d_input_Dec, d_out=d_target, widths=[16, 32, 16], dropout=0)
        self.FlagsPrediction = FeedForward(d_in=d_input_Dec, d_out=num_flags, widths=[32, 8], dropout=0)

        # Ce vecteur a pour but de déterminer l'action a réaliser, mettre fin à la traduction dans notre cas particulier
        self.ActionPrediction = FeedForward(d_in=d_input_Dec, d_out=1, widths=[32, 8], dropout=0)

        if self.FPIC:
            # Il faut donner à ce classifier l'impulsion fictive transportée dans l'espace des PDWs
            self.PulseToPDWs = FeedForward(d_in=d_input_Enc, d_out=d_input_Dec, widths=[], dropout=0)

        self.to(device)


    def forward(self, source, target):
        # target.shape = (batch_size, self.target_len, d_target+num_flags)
        batch_size, len_source, d_source = source.shape

        trg = self.TargetEmbedding(target)
        # trg.shape = (batch_size, self.target_len, d_input_Dec)

        src = self.SourceEmbedding(source)
        if self.FPIC:
            src, FakePulse = torch.split(src, len_source - 1, dim=1)

        for Encoder in self.Encoders:
            src = Encoder(src)
        for Decoder in self.Decoders:
            trg = Decoder(target=trg, source=src)
        # trg.shape = (batch_size, target_len, d_input_Dec)

        if self.FPIC:
            Physic, Flags, Action = self.PhysicalPrediction(trg), self.FlagsPrediction(trg), self.ActionPrediction(trg - self.PulseToPDWs(FakePulse))
        else:
            Physic, Flags, Action = self.PhysicalPrediction(trg), self.FlagsPrediction(trg), self.ActionPrediction(trg)

        # Physic.shape = (batch_size, target_len, d_target)
        # Flags.shape = (batch_size, target_len, num_flags)
        # Action.shape = (batch_size, target_len, 1)
        PDW = torch.cat((Physic, Flags), dim=2)
        return PDW, Action

    def translate(self, source):
        NbPDWsMemory = int(self.target_len / 2)
        Translations = []
        for sentence in source:
            Translation = []

            target = torch.zeros(size=(1, self.target_len, self.d_target + self.num_flags), device=self.device)
            for bursts in sentence:
                IdLastPDW = NbPDWsMemory
                Action = 1

                while Action > 0.5:
                    # Action=1 signifie qu'on publie le PDW d'indice IdLastPDW
                    with torch.no_grad():
                        target, Actions = self.forward(source=bursts.unsqueeze(0), target=target)
                    Action = Actions[0, IdLastPDW]
                    IdLastPDW += 1
                    if IdLastPDW == self.target_len - 1:
                        break
                        # raise ValueError("not enougth space to generate PDWs, fixe an higher target_len")

                # En sortant de la boucle, on attend la prochaine salve
                Translation += target[0, NbPDWsMemory:IdLastPDW].tolist()
                target = target[:, IdLastPDW-NbPDWsMemory:IdLastPDW]
                target = F.pad(target, (0, 0, 0, self.target_len-NbPDWsMemory))
            Translations.append(torch.tensor(Translation))

        return Translations