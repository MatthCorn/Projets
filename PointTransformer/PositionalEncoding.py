import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_pos: int, n_group: int):
        super().__init__()
        self.d_pos = d_pos

        self.add = nn.Sequential(
            nn.Linear(d_pos, 16), nn.ReLU(),
            nn.Linear(16, n_group), nn.ReLU(),
        )

        self.mult = nn.Sequential(
            nn.Linear(d_pos, 16), nn.ReLU(),
            nn.Linear(16, n_group), nn.ReLU(),
        )
