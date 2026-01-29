import torch.nn as nn

class Network(nn.Module):
    def __init__(
            self,
            n_layers,
            d_in=10,
            d_model=128,
            mem_length=5,
            dropout=0.0,
            **kwargs
    ):
        super().__init__()
        self.hidden_dim = d_model
        self.mem_length = mem_length

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)

        self.embedding = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_model, d_model),
        )

        self.post_rnn_ln = nn.LayerNorm(2 * d_model)

        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_model, d_in)
        )


    def forward(self, x):
        x = self.embedding(x)

        x, _ = self.lstm(x)  # [B, T, H]

        x = self.post_rnn_ln(x)

        x = self.head(x)  # [B, T, output_dim]

        return x
