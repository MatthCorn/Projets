import torch.nn as nn
from Transformer.RelativeMultiHeadSelfAttention import RMHSA
from Transformer.EasyFeedForward import FeedForward
from Perceiver.RelativeMultiHeadCrossAttention import RLCA

class EncoderLayer(nn.Module):
    def __init__(self, d_latent, d_input, d_att, num_heads, latent_len=16, WidthsFeedForward=[64], max_len=64, ADropout=0.1,
                 FFDropout=0.1, shared=True, masked=False):
        super().__init__()
        self.LatentLayerNorm1 = nn.LayerNorm(d_latent)
        self.InputLayerNorm1 = nn.LayerNorm(d_input)
        self.CrossAttentionLayer = RLCA(d_latent=d_latent, d_input=d_input, d_att=d_att, num_heads=num_heads, latent_len=latent_len, max_len=max_len, dropout=ADropout, masked=masked)
        self.LatentLayerNorm2 = nn.LayerNorm(d_latent)
        self.MLP1 = FeedForward(d_in=d_latent, d_out=d_latent, widths=WidthsFeedForward, dropout=FFDropout)
        self.LatentLayerNorm3 = nn.LayerNorm(d_latent)
        self.InputLayerNorm2 = nn.LayerNorm(d_input)
        self.SelfAttentionLayer = RMHSA(d_model=d_latent, d_att=d_att, num_heads=num_heads, max_len=max_len, dropout=ADropout, masked=masked)
        self.MLP2 = FeedForward(d_in=d_latent, d_out=d_latent, widths=WidthsFeedForward, dropout=FFDropout)

    def forward(self,x_input, x_latent):
        x_latent = self.CrossAttentionLayer(x_Latent=self.LatentLayerNorm1(x_latent), x_Input=self.InputLayerNorm1(x_input)) + x_latent
        x_latent = x_latent + self.MLP1(self.LatentLayerNorm2(x_latent))
        x_latent = self.SelfAttentionLayer(x_Latent=self.LatentLayerNorm3(x_latent), x_Input=self.InputLayerNorm2(x_input)) + x_latent
        x_latent = x_latent + self.MLP2(self.LatentLayerNorm2(x_latent))
        return x_latent

class EncoderIO(nn.Module):
    def __init__(self):
        super().__init__()
