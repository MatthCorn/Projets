import torch
from Transformer.EasyFeedForward import FeedForward
import torch.nn as nn

class TrackerEmbeddingLayer(nn.Module):

    def __init__(self, n_trackers, d_input_decoder, d_att, splitted=False):
        super().__init__()
        self.__splitted = splitted
        self.d_att = d_att
        if splitted:
            self.embeddings = nn.ModuleList()
            for i in range(n_trackers):
                self.embeddings.append(FeedForward(d_input_decoder, d_att, widths=[], dropout=0))
        else:
            self.Embedding = FeedForward(d_input_decoder, n_trackers*d_att, widths=[], dropout=0)

    def forward(self, input):
        batch_size, _, _ = input.shape
        if self.__splitted:
            y = tuple(Embedding(input) for Embedding in self.embeddings)
            y = torch.cat(y, dim=2)
        else:
            y = self.Embedding(input)
        y = y.reshape(batch_size, -1, self.d_att)
        return y
