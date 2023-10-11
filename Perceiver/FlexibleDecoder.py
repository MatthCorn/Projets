import torch.nn as nn
from Perceiver.RelativeMultiHeadCrossAttention import RLCA
from Transformer.EasyFeedForward import FeedForward

# Ce décodeur peut prendre des tailles de target différentes entre batchs, en utilisant en particulier RelativeMultiHeadCrossAttention.RLCA
# pour le self-attention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_att, num_heads, WidthsFeedForward=[512], RPR_len=64, target_len=64, MHADropout=0.1, FFDropout=0.1, masked=True, relative=True):
        super().__init__()
        self.RPR_len = RPR_len
        self.SelfAttentionLayer = RLCA(d_latent=d_model, d_input=d_model, d_att=d_att, num_heads=num_heads, latent_len=target_len,
                                       max_len=target_len, RPR_len=RPR_len, dropout=MHADropout, masked=masked, relative=relative)
        self.FirstLayerNorm = nn.LayerNorm(d_model)
        self.CrossAttentionLayer = RLCA(d_latent=d_model, d_input=d_model, d_att=d_att, num_heads=num_heads, dropout=MHADropout, masked=False, relative=False)
        self.SecondLayerNorm = nn.LayerNorm(d_model)
        self.FeedForward = FeedForward(d_model, d_model, widths=WidthsFeedForward, dropout=FFDropout)
        self.ThirdLayerNorm = nn.LayerNorm(d_model)

    def forward(self, target, source):
        y = self.FirstLayerNorm(self.SelfAttentionLayer(x_latent=target, x_input=target) + target)
        y = self.SecondLayerNorm(self.CrossAttentionLayer(x_latent=y, x_input=source) + y)
        y = self.ThirdLayerNorm(self.FeedForward(y) + y)
        return y

