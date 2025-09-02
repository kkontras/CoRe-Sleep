import torch.nn as nn
import torch
import numpy as np
import einops

class pos_encoding_sinusoidal(nn.Module):

    def __init__(self, d_hid, n_position=400):
        super(pos_encoding_sinusoidal, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

    def forward_concat(self, x):
        pos = self.pos_table[:, :x.size(1)].detach()
        x = torch.cat([ x, pos.repeat( x.shape[0], 1, 1)], dim=2)
        return x
class learned_pos_embedding(nn.Module):
    def __init__(self, max_pos, dim):
        super().__init__()
        self.pos_inner_tokens = nn.Parameter(torch.randn(max_pos, dim), requires_grad=True)
        self.pos_outer_tokens = nn.Parameter(torch.randn(max_pos, dim), requires_grad=True)

    def forward_outer(self, data):
        """

        :param data: [batch, outer, inner, mod, ch, features]
        :return: [batch, outer, inner, mod, ch, features] by having added pos_inner_tokens
        """

        data_shape = data.shape
        data = einops.rearrange(data, "b outer inner mod ch k -> (b inner mod ch) outer k")
        data = data + self.pos_inner_tokens[:data.shape[1],:].squeeze()
        data = einops.rearrange(data, "(b inner mod ch) outer k -> b outer inner mod ch k",  b=data_shape[0],  inner=data_shape[2], mod=data_shape[3], ch=data_shape[4])
        return data

    def forward_inner(self, data):
        """

        :param data: [batch, outer, inner, mod, ch, features]
        :return: [batch, outer, inner, mod, ch, features] by having added pos_outer_tokens
        """
        data_shape = data.shape
        data = einops.rearrange(data, "b outer inner mod ch k -> (b outer mod ch) inner k")
        data = data + self.pos_outer_tokens[:data.shape[1],:].squeeze()
        data = einops.rearrange(data, "(b outer mod ch) inner k -> b outer inner mod ch k",  b=data_shape[0],  outer=data_shape[1], mod=data_shape[3], ch=data_shape[4])
        return data
class sin_pos(nn.Module):
    def __init__(self, dmodel, npoints=400):
        super().__init__()
        self.pos = pos_encoding_sinusoidal(dmodel, n_position=npoints)

    def forward_inner(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod ch k -> (b outer mod ch) inner k")
        x = self.pos(x)
        x = einops.rearrange(x, "(b outer mod ch) inner k -> b outer inner mod ch k", b = self.batch, outer = self.outer, mod = self.mod, ch=self.ch)
        return x

    def forward_outer(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod ch k ->(b inner mod ch) outer k")
        x = self.pos(x)
        x = einops.rearrange(x, "(b inner mod ch) outer k -> b outer inner mod ch k", b = self.batch, inner = self.inner, mod = self.mod, ch = self.ch)
        return x

    def forward_time_inner(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.time, self.ch  = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        x = einops.rearrange(x, "b outer time ch k -> (b outer ch) time k")
        x = self.pos(x)
        x = einops.rearrange(x, "(b outer ch) time k-> b outer time ch k", b = self.batch, outer = self.outer, mod = self.mod, ch=self.ch)
        return x
class relative_pos_embedding(nn.Module):
    def __init__(self, tokens, dim_head, max_tokens = 2000, heads=None):
        """
        Output: [batch head tokens tokens]
        Args:
            tokens: the number of the tokens of the seq
            dim_head: the size of the last dimension of q
            heads: if None representation is shared across heads.
            else the number of heads must be provided
        """
        super().__init__()
        scale = dim_head ** -0.5
        self.shared_heads = heads if heads is not None else True
        if self.shared_heads:
            self.rel_pos_emb = nn.Parameter(torch.randn(heads, 2 * tokens - 1, dim_head) * scale)
        else:
            self.rel_pos_emb = nn.Parameter(torch.randn(2 * tokens - 1, dim_head) * scale)

        #Create an indice table to call the matrix and speed up during training
        indices = torch.arange(-tokens, tokens+1)
        self.indices_ext = torch.cat([indices[0].repeat(int((max_tokens - 2 * tokens + 1))), indices, indices[-1].repeat(int((max_tokens - 2 * tokens + 1)))], dim=0)
        self.indices_ext = self.indices_ext.unsqueeze(dim=0).repeat(max_tokens, 1)
        for i in range(max_tokens):
            self.indices_ext[i] = torch.roll(self.indices_ext[i], shifts=i)
        self.indices_ext = self.indices_ext[:max_tokens - tokens + 2, max_tokens - tokens + 1:]

    def forward(self, q):
        if self.shared_heads:
            this_indices_ext = self.indices_ext[:q.shape[2], :q.shape[2]]
            emb = torch.einsum('b h t d, h r t d -> b h t r', q, self.rel_pos_emb[:,this_indices_ext])
        else:
            this_indices_ext = self.indices_ext[:q.shape[2],q.shape[2]]
            emb = torch.einsum('b h t d, r t d -> b h t r', q, self.rel_pos_emb[this_indices_ext])

        return emb
