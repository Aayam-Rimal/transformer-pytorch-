import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):


    def __init__(self, d_model, max_len=5000):

        super().__init__()

        self.d_model= d_model

        pos= torch.arange(max_len).unsqueeze(1)
        i= torch.arange(d_model).unsqueeze(0)

        div_term = 1 / (10000 ** (2 * (i // 2) / d_model))
        angles = pos * div_term             

        pe = torch.zeros_like(angles)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]