import torch
import torch.nn as nn
import math
from torch.autograd.variable import Variable
import numpy as np

class PositionalEncoder(nn.Module):
    """
    Positional Encoder used in the paper "Attention is all you need"
    """
    def __init__(self, d_model, max_seq_len=21):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(int(max_seq_len)):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i == d_model -1 :
                    break
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # batch x seq x dmodel
        # make embeddings relatively larger
        x_shape = x.shape
        if len(x_shape)>3 :
            x = x.flatten(start_dim=2)
            x = x.permute(0,2,1)
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        return x.view(x_shape)

class Fourier_Sleep_PositionalEncoder(nn.Module):
    """
    Positional Encoder used in the paper "Attention is all you need"
    """
    def __init__(self, d_model, max_outer_seq_len=21, max_inner_seq_len=20, modalities=3):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i

        pe = torch.zeros(max_outer_seq_len, max_inner_seq_len, modalities, d_model)
        for pos_outer in range(int(max_outer_seq_len)):
            for pos_inner in range(int(max_inner_seq_len)):
                for pos_mod in range(int(modalities)):
                    for i in range(0, d_model, 2):
                        pe[pos_outer, pos_inner, pos_mod , i] = math.sin((pos_outer*(21**0)+pos_inner*(21**1)+pos_mod*(21**2)) / (100000 ** ((2 * i) / d_model)))
                        if i == d_model -1 :
                            break
                        pe[pos_outer, pos_inner, pos_mod , i+1] = math.cos((pos_outer*(21**0)+pos_inner*(21**1)+pos_mod*(21**2)) / (100000 ** ((2 * (i + 1)) / d_model)))

        # pe = nn.Parameter(torch.rand(max_outer_seq_len, max_inner_seq_len, modalities, d_model))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        # batch x outer x inner x mod x dmodel
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        x = x + Variable(self.pe[:,:self.outer, :self.inner, :self.mod, :],requires_grad=False).cuda()
        return x.view(x_shape)

class Fourier_Sleep_PositionalEncoder_Outer(nn.Module):
    """
    Positional Encoder used in the paper "Attention is all you need"
    """
    def __init__(self, d_model, max_outer_seq_len=21, modalities=3):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i

        # pe = torch.zeros(max_outer_seq_len, max_inner_seq_len, modalities, d_model)
        # for pos_outer in range(int(max_outer_seq_len)):
        #     for pos_inner in range(int(max_inner_seq_len)):
        #         for pos_mod in range(int(modalities)):
        #             for i in range(0, d_model, 2):
        #                 pe[pos_outer, pos_inner, pos_mod , i] = math.sin(pos_inner*pos_mod / (10000 ** ((2 * i) / d_model)))
        #                 if i == d_model -1 :
        #                     break
        #                 pe[pos_outer, pos_inner, pos_mod , i+1] = math.cos(pos_inner*pos_mod / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = nn.Parameter(torch.rand(max_outer_seq_len, 1, modalities, d_model))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        # batch x outer x inner x mod x dmodel
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        x = x + Variable(self.pe[:self.outer, 1, self.mod, :]).cuda()
        return x.view(x_shape)

class PositionalEncoding_AIAYN(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding_AIAYN, self).__init__()

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