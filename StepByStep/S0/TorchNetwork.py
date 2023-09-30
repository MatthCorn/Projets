from torch import nn, Tensor, Optional, Union, device, dtype
import torch
import math

class Network(nn.Transformer):

    def __init__(
            self,
            d_source=5,
            d_target=5,
            d_model=512,
            max_len=10,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.,
            activation='relu',
            batch_first=True,
            norm_first=False,
            device=torch.device('cpu')
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first)

        self.PE = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.source_embedding = nn.Linear(d_source, d_model)
        self.target_embedding = nn.Linear(d_target, d_model)
        self.output_dembedding = nn.Linear(d_model, d_target)
        self.device = device
        self.to(device)

    def forward(self, source: Tensor, target: Tensor, masked=True) -> Tensor:
        src, tgt = self.source_embedding(source), self.target_embedding(target)
        if masked:
            try:
                output = super().forward(src=self.PE(src), tgt=self.PE(tgt), tgt_mask=self.mask)
            except:
                self.generate_square_subsequent_mask(size=len(target[0]))
                output = super().forward(src=self.PE(src), tgt=self.PE(tgt), tgt_mask=self.mask)
        else:
            output = super().forward(src=self.PE(src), tgt=self.PE(tgt))

        return self.output_dembedding(output)

    def generate_square_subsequent_mask(self, size=200):  # Generate mask covering the top right triangle of a matrix
        self.mask = torch.triu(torch.full((size, size), float('-inf'), device=self.device), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


if __name__ == '__main__':
    N = Network(d_model=512, d_source=5, d_target=5, max_len=10, nhead=8, num_decoder_layers=3, num_encoder_layers=3)
    source = torch.rand(1000, 10, 5)
    target = torch.rand(1000, 10, 5)
    output = N(source=source, target=target, masked=True)
    output = N(source=source, target=torch.rand(1000, 9, 5), masked=True)

    print(output.shape)