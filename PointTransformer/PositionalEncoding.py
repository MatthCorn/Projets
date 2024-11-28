import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_pos: int):
        super().__init__()

        self.add = nn.Sequential(
            nn.Linear(d_pos, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.ReLU()
        )

        self.mult = nn.Sequential(
            nn.Linear(d_pos, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.ReLU()
        )
