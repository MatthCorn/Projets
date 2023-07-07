import torch.nn as nn
from Transformer.RelativeMultiHeadSelfAttention import RMHSA
from Perceiver.RelativeMultiHeadCrossAttention import RLCA
from Transformer.EasyFeedForward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_att, num_heads, WidthsFeedForward=[512], max_len=64, MHADropout=0.1, FFDropout=0.1, masked=True, relative=True):
        super().__init__()
        self.MultiHeadAttention = RMHSA(d_model=d_model, d_att=d_att, num_heads=num_heads, max_len=max_len, dropout=MHADropout, masked=masked, relative=relative)
        self.FirstLayerNorm = nn.LayerNorm(d_model)
        self.CrossAttentionLayer = RLCA(d_latent=d_model, d_input=d_model, d_att=d_att, num_heads=num_heads, latent_len=max_len,
                                        max_len=max_len, dropout=MHADropout, masked=False, relative=False)
        self.SecondLayerNorm = nn.LayerNorm(d_model)
        self.FeedForward = FeedForward(d_model, d_model, widths=WidthsFeedForward, dropout=FFDropout)
        self.ThirdLayerNorm = nn.LayerNorm(d_model)

    def forward(self, target, source):
        y = self.FirstLayerNorm(self.MultiHeadAttention(target) + target)
        y = self.SecondLayerNorm(self.CrossAttentionLayer(x_latent=y, x_input=source) + y)
        y = self.ThirdLayerNorm(self.FeedForward(y) + y)
        return y

if __name__ == '__main__':
    import torch

    DL = DecoderLayer(10, 10, 2)

    trg = torch.randn(3, 6, 10)
    # trg.shape = (batch_size, target_len, target_dim)
    src = torch.randn(3, 15, 10)
    # src.shape = (batch_size, source_len, source_dim)

    out = DL.forward(target=trg, source=src)
