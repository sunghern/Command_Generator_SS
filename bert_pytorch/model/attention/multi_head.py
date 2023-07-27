import time
import torch
import torch.nn as nn
from .single import Attention
import pim_cpp

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        start = time.time()
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        end = time.time()
        print(f"{end - start:.5f} attention sec")
        return self.output_linear(x)
 
class PIMMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.dim = d_model
        self.ffnw_1 = torch.rand(d_model, d_model * 4)
        self.ffnw_2 = torch.rand(d_model * 4, d_model)
        self.__set_qkv_weight_bias()

        self.output_linear = nn.Linear(d_model, d_model)

    def __set_qkv_weight_bias(self):
        self.qkv_weight = torch.rand([self.dim, self.dim * 3])

    def forward(self, query, key, value, mask=None):
        input = query.squeeze()
        len = input.size(0)
        output = pim_cpp.pim_msa(input, self.qkv_weight, self.ffnw_1, self.ffnw_2, len, self.dim)

        return output