import torch
import torch.nn as nn
import pim_cpp
from .gelu import GELU


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class PIMPositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PIMPositionwiseFeedForward, self).__init__()
        self.dim = d_model
    def forward(self, x):

        input = x.squeeze()
        len = input.size(0)
        output = pim_cpp.pim_ffn(input, len, self.dim)
        return output
