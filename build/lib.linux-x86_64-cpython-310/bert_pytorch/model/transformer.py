import torch.nn as nn

from .attention import PIMMultiHeadedAttention
from .utils import SublayerConnection, PIMPositionwiseFeedForward


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
        self.attention = PIMMultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PIMPositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        self.attention.forward(x, x, x, mask=mask)
        print("attention finish")
        print("attention finish", self.attention.forward(x, x, x, mask=mask).size())
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        print("layernorm finish")
        x = self.output_sublayer(x, self.feed_forward)
        print("FFN & layernorm finish")
        return self.dropout(x)