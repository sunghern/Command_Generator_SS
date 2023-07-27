import torch.nn as nn
import time

from .attention import PIMMultiHeadedAttention
from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PIMPositionwiseFeedForward
from .utils import PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention_cpu = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.attention = PIMMultiHeadedAttention(h=attn_heads, d_model=hidden)

        self.feed_forward = PIMPositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.feed_forward_cpu = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)

        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # start = time.time()
        # self.attention.forward(x, x, x, mask=mask)
        # end = time.time()
        # print("PIM attention time ", end-start)

        start = time.time()
        self.attention_cpu.forward(x, x, x, mask=mask)
        end = time.time()
        print("CPU attention time ", end-start)

        # start = time.time()
        # x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        # end = time.time()
        # print("PIM attention + layernorm time ", end-start)

        start = time.time()
        x = self.input_sublayer(x, lambda _x: self.attention_cpu.forward(_x, _x, _x, mask=mask))
        end = time.time()
        print("CPU attention + layernorm time ", end-start)

        # start = time.time()
        # self.feed_forward
        # end = time.time()
        # print("PIM FFN time ", end-start)

        start = time.time()
        self.feed_forward_cpu
        end = time.time()
        print("CPU FFN time ", end-start)

        # start = time.time()
        # x = self.output_sublayer(x, self.feed_forward)
        # end = time.time()
        # print("PIM FFN + layernorm time ", end-start)

        start = time.time()
        x = self.output_sublayer(x, self.feed_forward_cpu)
        end = time.time()
        print("CPU FFN + layernorm time ", end-start)

        print("FFN & layernorm finish")
        return self.dropout(x)
