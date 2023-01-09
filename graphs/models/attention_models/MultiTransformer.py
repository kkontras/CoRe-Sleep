import einops
from graphs.models.attention_models.utils.positionalEncoders import *
from graphs.models.attention_models.seq_base_models.mTransformerEncoder import Transformer_Encoder, myTransformerEncoderLayer
import torch.nn as nn
from graphs.models.attention_models.ViLBERT import MyViLBERT
from graphs.models.custom_layers.attention import *
from typing import Optional, Any
from torch import Tensor


class TF_inner_mod_att_ch_diff_fc(nn.Module):
    def __init__(self, d_model, nhead, modalities=1, dim_feedforward=2048, dropout=0.1, activation="relu"):
        # super(nn.TransformerEncoderLayer, self).__init__()
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Implementation of Feedforward model
        for i in range(modalities):
            setattr(self,"mod_{}_linear1".format(i),nn.Linear(d_model, dim_feedforward))
            setattr(self,"mod_{}_dropout".format(i),nn.Dropout(dropout))
            setattr(self,"mod_{}_linear2".format(i),nn.Linear(dim_feedforward, d_model))
            setattr(self,"mod_{}_norm2".format(i),nn.LayerNorm(d_model))
            setattr(self,"mod_{}_dropout2".format(i),nn.Dropout(dropout))

        self.activation = nn.ReLU()

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x_shape = src.shape

        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        src = einops.rearrange(src, "b outer inner mod ch k -> (inner mod) (b outer) (ch k)")

        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = einops.rearrange(src, "(inner mod) b_outer ch_k -> mod inner b_outer ch_k", inner = self.inner, mod = self.mod)

        for i in range(len(src)):
            lin1 = getattr(self,"mod_{}_linear1".format(i))
            drop = getattr(self,"mod_{}_dropout".format(i))
            lin2 = getattr(self,"mod_{}_linear2".format(i))
            norm2 = getattr(self,"mod_{}_norm2".format(i))
            drop2 = getattr(self,"mod_{}_dropout2".format(i))
            src2 = lin2(drop(self.activation(lin1(src[i]))))
            src[i] = src[i] + drop2(src2)
            src[i] = norm2(src[i])

        src = einops.rearrange(src, "mod inner (b outer) (ch k) -> b outer inner mod ch k",b=self.batch, outer=self.outer, ch=self.ch,
                             mod=self.mod)
        return src

class TF_outer_mod_att_inner_ch_diff_fc(nn.Module):
    def __init__(self, d_model, nhead, modalities=1, dim_feedforward=2048, dropout=0.1, activation="relu"):
        # super(nn.TransformerEncoderLayer, self).__init__()
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Implementation of Feedforward model
        for i in range(modalities):
            setattr(self,"mod_{}_linear1".format(i),nn.Linear(d_model, dim_feedforward))
            setattr(self,"mod_{}_dropout".format(i),nn.Dropout(dropout))
            setattr(self,"mod_{}_linear2".format(i),nn.Linear(dim_feedforward, d_model))
            setattr(self,"mod_{}_norm2".format(i),nn.LayerNorm(d_model))
            setattr(self,"mod_{}_dropout2".format(i),nn.Dropout(dropout))

        self.activation = nn.ReLU()

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x_shape = src.shape

        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        src = einops.rearrange(src, "b outer inner mod ch k -> (outer mod) b (inner ch k)")

        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = einops.rearrange(src, "(outer mod) b inner_ch_k -> mod outer b inner_ch_k", outer = self.outer, mod = self.mod)

        for i in range(len(src)):
            lin1 = getattr(self,"mod_{}_linear1".format(i))
            drop = getattr(self,"mod_{}_dropout".format(i))
            lin2 = getattr(self,"mod_{}_linear2".format(i))
            norm2 = getattr(self,"mod_{}_norm2".format(i))
            drop2 = getattr(self,"mod_{}_dropout2".format(i))
            src2 = lin2(drop(self.activation(lin1(src[i]))))
            src[i] = src[i] + drop2(src2)
            src[i] = norm2(src[i])

        src = einops.rearrange(src, "mod outer b (inner ch k) -> b outer inner mod ch k",b=self.batch, inner=self.inner, ch=self.ch,
                             mod=self.mod)
        return src


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden):
        # timestep = hidden.size(0)
        # h = hidden.repeεat(timestep, 1, 1).transpose(0, 1)
        # print(h.shape)
        hidden = hidden.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(hidden, hidden)
        return self.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]
class Context_Attention(nn.Module):
    def __init__(self, hidden, attention_size=64):
        super().__init__()
        self.attention_size = attention_size

        self.attn = nn.Linear(hidden, attention_size)
        self.ae = nn.Parameter(torch.rand(attention_size))
        self.softmax = nn.Softmax(dim=1)
        self.tanh= nn.Tanh()

    def forward(self, x):
        # batch
        x = self.tanh(self.attn(x))
        ae = self.ae.repeat(x.size(0), 1).unsqueeze(1)
        x = einsum("bij, bmj-> bi",x,ae)
        x = self.softmax(x)
        return x

class inner_mod_att_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)
            self.pos_mod = PositionalEncoder(d_model=dmodel)

        enc = nn.TransformerEncoderLayer(dmodel, nhead=heads)
        self.inner_tf = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod) (b outer) (ch k)")
        if self.pos:
            x = einops.rearrange(x, "(inner mod) (b outer) k -> inner (b outer mod) k", mod=self.mod, outer=self.outer,
                                 b=self.batch, inner=self.inner)
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "inner (b outer mod) k -> mod (b outer inner) k", mod=self.mod, outer=self.outer,
                                 b=self.batch, inner=self.inner)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "mod (b outer inner) k -> (inner mod) (b outer) k", mod=self.mod, outer=self.outer,
                                 b=self.batch, inner=self.inner)
        x = self.inner_tf(x)
        x = einops.rearrange(x, "(inner mod) (b outer) (ch k) -> b outer inner mod ch k", mod=self.mod, ch=self.ch,
                             outer=self.outer, b=self.batch, inner=self.inner)
        return x
class inner_mod_ch_att(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)
            self.pos_mod = PositionalEncoder(d_model=dmodel)

        enc = nn.TransformerEncoderLayer(dmodel, nhead=heads)
        self.inner_tf = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        if self.pos:
            x = einops.rearrange(x, "(inner mod) (b outer) k -> inner (b outer mod) k", mod=self.mod, outer=self.outer,
                                 b=self.batch, inner=self.inner)
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "inner (b outer mod) k -> mod (b outer inner) k", mod=self.mod, outer=self.outer,
                                 b=self.batch, inner=self.inner)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "mod (b outer inner) k -> (inner mod) (b outer) k", mod=self.mod, outer=self.outer,
                                 b=self.batch, inner=self.inner)
        x = self.inner_tf(x)
        x = einops.rearrange(x, "(inner mod ch) (b outer) k -> b outer inner mod ch k", mod=self.mod, ch=self.ch,
                             outer=self.outer, b=self.batch, inner=self.inner)
        return x
class inner_mod_outer_att_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)
            self.pos_mod = PositionalEncoder(d_model=dmodel)

        enc = nn.TransformerEncoderLayer(dmodel, nhead=heads)
        self.all_tf = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        if self.pos:
            x = einops.rearrange(x, "b outer inner mod k -> inner (b outer mod) k")
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "inner (b outer mod) k -> mod (b outer inner) k", mod=self.mod)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "mod (b outer inner) k -> outer (b inner mod) k", outer=self.outer)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "outer (b inner mod) k -> b outer inner mod k", mod=self.mod, inner=self.inner,
                                 b=self.b)

        x = einops.rearrange(x, "b outer inner mod ch k -> (outer inner mod) b (ch k)")
        x = self.all_tf(x)
        x = einops.rearrange(x, "(outer inner mod) b (ch k) -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,
                             b=self.batch)
        return x
class inner_att_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=1024)
        self.inner_tf = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> inner (b mod outer) (ch k)")
        if self.pos:
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_tf(x)
        x = einops.rearrange(x, "inner (b mod outer) (ch k) -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,
                             b=self.batch)
        return x
class inner_mod_att_ch_diff_FC(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = TF_inner_mod_att_ch_diff_fc(d_model=dmodel, modalities=modalities, nhead=heads, dim_feedforward=1024)
        self.inner_tf = nn.TransformerEncoder(enc, num_layers)


    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = self.inner_tf(x)

        return x


class inner_att_mod_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel * modalities)

        enc = nn.TransformerEncoderLayer(dmodel * modalities, nhead=heads)
        self.inner_tf = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod k -> inner (b outer) (mod ch k)")
        if self.pos:
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_tf(x)
        x = einops.rearrange(x, "inner (b outer) (mod ch k) -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,
                             b=self.batch)
        return x
class inner_att_outer_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel * outer)

        enc = nn.TransformerEncoderLayer(dmodel * outer, nhead=heads)
        self.inner_tf = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> inner (mod b) (outer ch k)")
        if self.pos:
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_tf(x)
        x = einops.rearrange(x, "inner (mod b) (outer ch k) -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,
                             b=self.batch)
        return x
class inner_att_mod_ch_outer(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel * modalities * outer)

        enc = nn.TransformerEncoderLayer(dmodel * modalities * outer, nhead=heads)
        self.inner_tf = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> inner b (outer mod ch k)")
        if self.pos:
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_tf(x)
        x = einops.rearrange(x, " inner b (outer mod ch k) -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,
                             b=self.batch)
        return x

class inner_outer_cross_att_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)
            self.pos_mod = PositionalEncoder(d_model=dmodel)

        self.num_layers = num_layers
        for i in range(num_layers):
            setattr(self, "outter_inner_cross_att_{}".format(i), MyViLBERT(dmodel, nheads=heads))

        print("Inner_outer_cross_att only supports two modalities for the moment!")

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        if self.pos:
            x = einops.rearrange(x, "b outer inner mod k -> inner (b outer mod) k")
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "inner (b outer mod) k -> mod (b outer inner) k", mod=self.mod)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "mod (b outer inner) k -> outer (b inner mod) k", outer=self.outer)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "outer (b inner mod) k -> b outer inner mod k", mod=self.mod, inner=self.inner,
                                 b=self.b)

        x = einops.rearrange(x, "b outer inner mod ch k -> (outer inner) mod b (ch k)")
        x0 = x[:, 0, :, :]
        x1 = x[:, 1, :, :]
        for i in range(self.num_layers):
            layer = getattr(self, "outter_inner_cross_att_{}".format(i))
            x0, x1 = layer(x0, x1)
        x0 = einops.rearrange(x0, "(outer inner mod) b k -> (outer inner) mod b k", outer=self.outer, inner=self.inner,
                              mod=1)
        x1 = einops.rearrange(x1, "(outer inner mod) b k -> (outer inner) mod b k", outer=self.outer, inner=self.inner,
                              mod=1)
        x = torch.cat([x0, x1], dim=1)
        x = einops.rearrange(x, "(outer inner) mod b (ch k) -> b outer inner mod ch k", outer=self.outer, mod=self.mod,
                             ch=self.ch,
                             b=self.batch)
        return x
class inner_cross_att_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)
            self.pos_mod = PositionalEncoder(d_model=dmodel)

        self.num_layers = num_layers
        for i in range(num_layers):
            setattr(self, "inner_cross_att_{}".format(i), MyViLBERT(dmodel, nheads=heads))

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        if self.pos:
            x = einops.rearrange(x, "b outer inner mod k -> inner (b outer mod) k")
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "inner (b outer mod) k -> mod (b outer inner) k", mod=self.mod)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "mod (b outer inner) k -> outer (b inner mod) k", outer=self.outer)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "outer (b inner mod) k -> b outer inner mod k", mod=self.mod, inner=self.inner,
                                 b=self.b)

        x = einops.rearrange(x, "b outer inner mod ch k -> inner mod (outer b) (ch k)")
        x0 = x[:, 0, :, :]
        x1 = x[:, 1, :, :]
        for i in range(self.num_layers):
            layer = getattr(self, "inner_cross_att_{}".format(i))
            x0, x1 = layer(x0, x1)
        x0 = einops.rearrange(x0, "(inner mod) b k -> inner mod b k", inner=self.inner, mod=1)
        x1 = einops.rearrange(x1, "(inner mod) b k -> inner mod b k", inner=self.inner, mod=1)
        x = torch.cat([x0, x1], dim=1)
        x = einops.rearrange(x, "inner mod (outer b) (ch k) -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,
                             b=self.batch)
        return x
class outer_cross_att_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=dmodel)
            self.pos_mod = PositionalEncoder(d_model=dmodel)
        self.num_layers = num_layers
        for i in range(num_layers):
            setattr(self, "outter_cross_att_{}".format(i), MyViLBERT(dmodel, nheads=heads))

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        if self.pos:
            x = einops.rearrange(x, "b outer inner mod k -> outer (b outer mod) k")
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "inner (b outer mod) k -> mod (b outer inner) k", mod=self.mod)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "mod (b outer inner) k -> outer (b inner mod) k", outer=self.outer)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "outer (b inner mod) k -> b outer inner mod k", mod=self.mod, inner=self.inner,
                                 b=self.b)

        x = einops.rearrange(x, "b outer inner mod ch k -> outer mod (inner b) (ch k)")
        x0 = x[:, 0, :, :]
        x1 = x[:, 1, :, :]
        for i in range(self.num_layers):
            layer = getattr(self, "outter_cross_att_{}".format(i))
            x0, x1 = layer(x0, x1)
        x0 = einops.rearrange(x0, "(outer mod) b k -> outer mod b k", outer=self.outer, mod=1)
        x1 = einops.rearrange(x1, "(outer mod) b k -> outer mod b k", outer=self.outer, mod=1)
        x = torch.cat([x0, x1], dim=1)
        x = einops.rearrange(x, "outer mod (inner b) (ch k) -> b outer inner mod ch k", inner=self.inner, mod=self.mod, ch=self.ch,
                             b=self.batch)
        return x

class mod_att_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_mod = PositionalEncoder(d_model=dmodel)

        enc_mod = nn.TransformerEncoderLayer(dmodel, nhead=heads)
        self.inner_mod_tf = nn.TransformerEncoder(enc_mod, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> mod (inner outer b) (ch k)")
        if self.pos:
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_mod_tf(x)
        x = einops.rearrange(x, "mod (inner outer b) (ch k)-> b outer inner mod ch k", outer=self.outer, inner=self.inner, ch=self.ch,
                             b=self.batch)
        return x
class mod_att_inner_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_mod = PositionalEncoder(d_model=dmodel * inner)

        enc_mod = nn.TransformerEncoderLayer(dmodel * inner, nhead=heads)
        self.inner_mod_tf = nn.TransformerEncoder(enc_mod, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> mod (outer b) (inner ch k)")
        if self.pos:
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_mod_tf(x)
        x = einops.rearrange(x, " mod (outer b) (inner ch k)-> b outer inner mod ch k", outer=self.outer, inner=self.inner, ch=self.ch,
                             b=self.batch)
        return x
class mod_att_outer_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_mod = PositionalEncoder(d_model=dmodel * outer)

        enc_mod = nn.TransformerEncoderLayer(dmodel * outer, nhead=heads)
        self.inner_mod_tf = nn.TransformerEncoder(enc_mod, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> mod (inner b) (outer ch k)")
        if self.pos:
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_mod_tf(x)
        x = einops.rearrange(x, "mod (inner b) (outer ch k)-> b outer inner mod ch k", outer=self.outer, inner=self.inner, ch=self.ch,
                             b=self.batch)
        return x
class mod_att_inner_outer_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_mod = PositionalEncoder(d_model=dmodel * inner * outer)

        enc_mod = nn.TransformerEncoderLayer(dmodel * inner * outer, nhead=heads)
        self.inner_mod_tf = nn.TransformerEncoder(enc_mod, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> mod b (inner outer ch k)")
        if self.pos:
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_mod_tf(x)
        x = einops.rearrange(x, "mod b (inner outer ch k)-> b outer inner mod ch k", outer=self.outer, inner=self.inner, ch=self.ch,
                             b=self.batch)
        return x

class outer_att_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=dmodel)

        enc_outer = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=1024)
        self.outer_tf = nn.TransformerEncoder(enc_outer, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> outer (b inner mod) (ch k)")
        if self.pos:
            x = self.pos_outer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.outer_tf(x)
        x = einops.rearrange(x, "outer (b inner mod) (ch k)-> b outer inner mod ch k", mod=self.mod, inner=self.inner, ch=self.ch,
                             b=self.batch)
        return x
class outer_att_mod_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=dmodel * modalities)

        enc_outer = nn.TransformerEncoderLayer(dmodel * modalities, nhead=heads)
        self.outer_tf = nn.TransformerEncoder(enc_outer, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> outer (b inner) (mod ch k)")
        if self.pos:
            x = self.pos_outer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.outer_tf(x)
        x = einops.rearrange(x, "outer (b inner) (mod ch k)-> b outer inner mod ch k", mod=self.mod, inner=self.inner, ch=self.ch,
                             b=self.batch)
        return x
class outer_att_inner_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=dmodel * inner)

        enc_outer = nn.TransformerEncoderLayer(dmodel * inner, nhead=heads)
        self.outer_tf = nn.TransformerEncoder(enc_outer, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> outer (b mod) (inner ch k)")
        if self.pos:
            x = self.pos_outer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.outer_tf(x)
        x = einops.rearrange(x, "outer (b mod) (inner ch k)-> b outer inner mod ch k", mod=self.mod, inner=self.inner, ch=self.ch,
                             b=self.batch)
        return x
class outer_att_inner_mod_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=dmodel * inner * modalities)

        enc_outer = nn.TransformerEncoderLayer(dmodel * inner * modalities, nhead=heads)
        self.outer_tf = nn.TransformerEncoder(enc_outer, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> outer b (inner mod ch k)")
        if self.pos:
            x = self.pos_outer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.outer_tf(x)
        x = einops.rearrange(x, "outer b (inner mod ch k) -> b outer inner mod ch k", mod=self.mod, inner=self.inner, ch=self.ch,
                             b=self.batch)
        return x
class outer_mod_att_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)
            self.pos_mod = PositionalEncoder(d_model=dmodel)

        enc = nn.TransformerEncoderLayer(dmodel, nhead=heads)
        self.all_tf = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        if self.pos:
            x = einops.rearrange(x, "b outer inner mod k -> inner (outer mod b) k")
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "inner (b outer mod) k -> mod (b outer inner) k", mod=self.mod, outer=self.outer,
                                 b=self.batch, inner=self.inner)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "mod (b outer inner) k -> outer (b inner mod) k", mod=self.mod, outer=self.outer,
                                 b=self.batch, inner=self.inner)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "outer (inner mod b) k -> b outer inner mod k", mod=self.mod, outer=self.outer,
                                 b=self.batch, inner=self.inner)

        x = einops.rearrange(x, "b outer inner mod ch k -> (outer mod) (b inner) (ch k)")
        x = self.all_tf(x)
        x = einops.rearrange(x, "(outer mod) (b inner) (ch k) -> b outer inner mod ch k", mod=self.mod, outer=self.outer, ch=self.ch,
                             b=self.batch, inner=self.inner)
        return x
class outer_mod_att_inner_ch_diff_FC(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = TF_outer_mod_att_inner_ch_diff_fc(d_model=dmodel, modalities=modalities, nhead=heads, dim_feedforward=1024)
        self.tf = nn.TransformerEncoder(enc, num_layers)


    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]
        x = self.tf(x)

        return x

class aggregation_att_outer_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Attention(dmodel)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> outer (b inner mod) (ch k) ", mod=self.mod, inner=self.inner,
                             b=self.batch)
        w = self.mod_att(x)
        x = torch.einsum("ijk,jmi -> mjk", x, w)
        x = einops.rearrange(x, " outer (b inner mod) (ch k)  -> b outer inner mod ch k ", b=self.batch, inner=self.inner, ch=self.ch,
                             mod=self.mod)
        return x
class aggregation_att_inner_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Attention(dmodel * modalities)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> inner (b outer) (mod ch k) ", mod=self.mod, inner=self.inner,
                             b=self.batch)
        w = self.mod_att(x)
        x = torch.einsum("ijk,jmi -> mjk", x, w)
        x = einops.rearrange(x, "inner (b outer mod) (ch k)  -> b outer inner mod ch k ", b=self.batch, outer=self.outer, ch=self.ch,
                             mod=self.mod)
        return x
class aggregation_att_contx_inner_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Context_Attention(dmodel * modalities, 64)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> (b outer) inner (mod ch k) ", mod=self.mod, inner=self.inner,
                             b=self.batch)

        w = self.mod_att(x)

        x = torch.einsum("ijk,im -> ik", x, w)
        x = einops.rearrange(x, "(b outer inner) (mod ch k)  -> b outer inner mod ch k ", b=self.batch, outer=self.outer, ch=self.ch,
                             mod=self.mod, inner=1)
        return x
class aggregation_att_contx_inner_mod_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Context_Attention(dmodel, 64)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> (b outer) (inner mod) (ch k)", mod=self.mod, inner=self.inner,
                             b=self.batch)
        w = self.mod_att(x)
        x = torch.einsum("ijk,im -> ik", x, w)
        x = einops.rearrange(x, "(b outer inner mod) (ch k)  -> b outer inner mod ch k ", b=self.batch, outer=self.outer, mod=1, ch=self.ch,
                             inner=1)
        return x
class aggregation_att_contx_mod_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Context_Attention(dmodel, 64)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> (b outer inner) mod (ch k) ", mod=self.mod, inner=self.inner,
                             b=self.batch)
        w = self.mod_att(x)
        x = torch.einsum("ijk,im -> ik", x, w)
        x = einops.rearrange(x, "(b outer inner mod) (ch k) -> b outer inner mod ch k ", b=self.batch, outer=self.outer, ch=self.ch,
                             mod=1, inner=self.inner)
        return x
class aggregation_att_mod_ch(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Attention(dmodel)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> mod (inner outer b) (ch k) ", mod=self.mod, inner=self.inner,
                             b=self.batch)
        w = self.mod_att(x)
        x = torch.einsum("ijk,jmi -> mjk", x, w)
        x = einops.rearrange(x, "mod (inner outer b) (ch k)  -> b outer inner mod ch k -> ", inner=self.inner, b=self.batch, ch=self.ch)
        return x

class fourier_pos(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = Fourier_Sleep_PositionalEncoder(dmodel, outer, inner, modalities)

    def forward(self, x):
        return self.pos(x)
class huy_pos_inner(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = PositionalEncoding_AIAYN(dmodel)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> (b outer mod) inner (ch k)")
        x = self.pos(x)
        x = einops.rearrange(x, "(b outer mod) inner (ch k) -> b outer inner mod ch k", b=self.batch, outer=self.outer, ch=self.ch,
                             mod=self.mod)
        return x
class huy_pos_outer(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = PositionalEncoding_AIAYN(dmodel)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k ->(b inner mod) outer (ch k)")
        x = self.pos(x)
        x = einops.rearrange(x, "(b inner mod) outer (ch k) -> b outer inner mod ch k", b=self.batch, inner=self.inner, ch=self.ch,
                             mod=self.mod)
        return x

class Multi_Transformer_v2(nn.Module):

    def __init__(self, dmodel, pos, inner, outer, layers=["inner_att", "outer_att"], modalities=1, channels=1,
                 num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        self.layers = layers
        for layer in self.layers:
            print(layer)
            setattr(self, layer, globals()[layer](dmodel, pos, inner, outer, modalities, channels, num_layers, heads))

    def forward(self, x):
        for layer in self.layers:
            this_layer = getattr(self, layer)
            x = this_layer(x)
        return x