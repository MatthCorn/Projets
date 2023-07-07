from Transformer import EasyFeedForward, EncoderTransformer
import torch
import torch.nn as nn

class TransformerTranslator(nn.Module):

    def __init__(self):
        super().__init__()
        *