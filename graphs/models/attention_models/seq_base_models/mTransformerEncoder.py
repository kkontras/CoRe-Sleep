import torch.nn as nn
from typing import Optional, Any
import torch
import torch.functional as F
from graphs.models.attention_models.utils.positionalEncoders import PositionalEncoder
"""
Transformer encoder to insert the different output merging methods
"""
class Transformer_Encoder (nn.Module):
    def __init__(self, d_model, nhead, num_layers, positional_encoder= None, merge_func = lambda x: x):

        super().__init__()

        if positional_encoder == None: positional_encoder = PositionalEncoder(d_model= d_model)

        self.pos = positional_encoder

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.tranf_encoder = nn.TransformerEncoder(encoder_layer, num_layers= num_layers)

        self.merge_func = merge_func

    def forward(self, x):

        x = self.pos(x)
        transf_out = self.tranf_encoder(x)
        return self.merge_func(transf_out)


class myTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(myTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.ReLU()
        super(myTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, query: torch.Tensor, key: torch.Tensor=None, value: torch.Tensor=None, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if key == None:
            key = query
        if value == None:
            value = query
        print(query.shape)
        print(key.shape)
        print(value.shape)
        src2 = self.self_attn(query, key, value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        print(src2.shape)
        value = value + self.dropout1(src2)
        value = self.norm1(value)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(value))))
        value = value + self.dropout2(src2)
        value = self.norm2(value)
        return value

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))