import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.functional as F
from graphs.models.attention_models.seq_base_models.mLSTM import LSTM
from graphs.models.attention_models.seq_base_models.mTransformerEncoder import Transformer_Encoder, myTransformerEncoderLayer
from graphs.models.attention_models.utils.positionalEncoders import *
import copy
from graphs.models.attention_models.stand_alone_att_vision import *
from graphs.models.attention_models.dynamic_cov import *
from graphs.models.custom_layers.attention import *
from graphs.models.custom_layers.MulilogueNet import *
from graphs.models.custom_layers.LSTHM import *
import einops
from graphs.models.attention_models.ViLBERT import MyViLBERT
from graphs.models.custom_layers.Transformer_Encoder_Huy import TransformerEncoderLayer_Huy
from typing import Optional, Any
from torch import Tensor
from joblib import Parallel, delayed
from tqdm import tqdm

class EEG_Encoder_E_3(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.pad_1 = nn.ReflectionPad2d((5,5,1,1))
        # self.pad_1 = nn.ZeroPad2d((5,5,1,1))
        self.conv1 = nn.Conv2d(1, 10*dec, kernel_size=(2, 10), stride=(1, 1))

        self.pad_2 = nn.ReflectionPad2d((2,2,0,0))
        # self.pad_2 = nn.ZeroPad2d((2,2,0,0))
        self.conv2 = nn.Conv2d(10*dec, 20*dec, kernel_size=(1, 5), stride=(1, 1))

        self.conv3 = nn.Conv2d(20*dec, 20*dec, kernel_size=(4, 1), stride=(1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 3))
        self.maxpool_time = nn.MaxPool2d(kernel_size=(1, 3))
        self.relu = torch.nn.ReLU()
        self.conv1_bn = nn.BatchNorm2d(1)

    def forward(self,x):
        x1 = self.relu(self.conv1(self.pad_1(self.conv1_bn(x))))
        x2 = self.maxpool(x1)
        x3 = self.relu(self.conv2(self.pad_2(x2)))
        x4 = self.relu(self.conv3(x3))
        x5 = self.maxpool_time(x4)
        return x5

class EEG_Encoder_Ch(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.pad_1 = nn.ReflectionPad1d(2)
        self.conv1 = nn.Conv1d(1, 20*dec, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(20*dec, 80*dec, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(80*dec, 160*dec, kernel_size=3, stride=1)
        self.conv4 = nn.Conv1d(160*dec, 80*dec, kernel_size=1, stride=1)
        self.conv5 = nn.Conv1d(80*dec, 40*dec, kernel_size=3, stride=1)
        self.conv6 = nn.Conv1d(40*dec, 20*dec, kernel_size=1, stride=1)

        self.pad_2 = nn.ReflectionPad1d(1)
        # self.pad_2 = nn.ZeroPad2d((2,2,0,0))
        # self.conv2 = nn.Conv1d(10*dec, 20*dec, kernel_size=5, stride=1)

        self.maxpool_time = nn.MaxPool1d(4)
        # self.maxpool_time = nn.AdaptiveAvgPool1d(225)
        self.mypool = nn.Conv1d(10*dec, 10*dec, kernel_size=4, stride=4)
        self.relu = torch.nn.ReLU()
        self.conv1_bn = nn.BatchNorm1d(1)
        self.conv2_bn = nn.BatchNorm1d(10*dec)

        # self.alpha = nn.Parameter(torch.randn(10*dec, 4),requires_grad=True)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.relu(self.conv1(self.pad_1(x)))
        x = self.relu(self.conv2(x))
        x = self.maxpool_time(x)

        x = self.relu(self.conv3(self.pad_2(x)))
        x = self.relu(self.conv4(x))
        x = self.maxpool_time(x)
        x = self.relu(self.conv5(self.pad_2(x)))
        x = self.relu(self.conv6(x))

        # x = self.maxpool_time(x)

        return x

class EEG_Encoder_TFN(nn.Module):
    def __init__(self, dec, d):
        super().__init__()
        self.conv_ch = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 64*dec, kernel_size=(1, 5), bias=False),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(64 * dec, 128 * dec, kernel_size=(1, 5), bias=False),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(128*dec, 2 * dec, kernel_size=(1, 5), bias=False),
            nn.ReLU(),
            nn.AvgPool2d((1,56))
        )
        self.conv_all = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 64*dec, kernel_size=(1, 5), bias=False),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
            nn.ReflectionPad2d((2, 2, 1, 1)),
            nn.Conv2d(64 * dec, 128 * dec, kernel_size=(2, 5), bias=False),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(128*dec, 14 * dec, kernel_size=(2, 5), bias=False),
            nn.ReLU(),
            nn.AvgPool2d((1,56))
        )


    def forward(self,x):
        x_all = self.conv_all(x)
        x_all = x_all.flatten(start_dim=1,end_dim=2).unsqueeze(dim=2)
        m = []
        for i in range(8):
            a =x[:,:,i,:]
            # print(a.shape)
            m.append(self.conv_ch(a.unsqueeze(dim=2)))
        m = torch.cat(m,dim=1)
        x = torch.cat([x_all,m],dim=1)
        return x

class EEG_Encoder_MultiToken(nn.Module):
    def __init__(self, dec, transformer_type):
        super().__init__()

        # self.maxpool = nn.MaxPool2d(kernel_size=(1, 4))
        # self.pos1 = PositionalEncoder(d_model=dmodel*8, same_time_step=7)

        dmodel = 32 * dec

        self.conv_eeg = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, dmodel, kernel_size=(2, 5), stride=(1, 2), bias=False),
            # nn.ReLU(),
        )
        self.conv_eog = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, dmodel, kernel_size=(1, 5), stride=(1, 2), bias=False),
            # nn.ReLU(),
        )

        self.interm_eeg_conv_0 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(dmodel, dmodel, kernel_size=(1, 5), stride=(1, 2), bias=False),
            # nn.ReLU(),
        )
        self.interm_eog_conv_0 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(dmodel, dmodel, kernel_size=(1, 5), stride=(1, 2), bias=False),
            # nn.ReLU(),
        )
        self.interm_eeg_conv_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 2, 0, 0)),
            nn.Conv2d(dmodel, dmodel, kernel_size=(1, 5), stride=(1, 2), bias=False),
            # nn.ReLU(),
        )
        self.interm_eog_conv_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 2, 0, 0)),
            nn.Conv2d(dmodel, dmodel, kernel_size=(1, 5), stride=(1, 2), bias=False),
            # nn.ReLU(),
        )
        # self.latent = nn.Parameter(torch.randn(16, 8, dmodel))
        self.cls_token = nn.Parameter(torch.randn(1, dmodel,1, 1))
        # self.cls_token = torch.cat([self.cls_token]*22, dim=0)
        self.pos = PositionalEncoder(d_model=dmodel, same_time_step=8)
        transformer_type = globals()[transformer_type]

        self.att1_eeg = transformer_type(dmodel, 1)
        self.att1_eog = transformer_type(dmodel, 1)
        # self.att1_1 = My_Transformer_Layer(dmodel*450)
        self.att2_eeg = transformer_type(dmodel, 1)
        self.att2_eog = transformer_type(dmodel, 1)
        # self.att2_1 = My_Transformer_Layer(dmodel*2*225)
        self.att3_eeg = transformer_type(dmodel, 1)
        self.att3_eog = transformer_type(dmodel, 1)

        # self.att3_1 = My_Transformer_Layer(16 * dec*112)
        self.avg_1 = nn.AvgPool2d(kernel_size=(1, 1500))
        self.avg_2 = nn.AvgPool2d(kernel_size=(1, 750))
        self.avg_3 = nn.AvgPool2d(kernel_size=(1, 375))
        # self.avg = nn.AvgPool2d(kernel_size=(1, 23))


    def forward(self,x):
        x_shape = x[0].shape
        if len(x_shape)>4:
            for i in range(len(x)):
                x[i] = x[i].flatten(start_dim=0, end_dim=1)

        xeeg = self.conv_eeg(x[0])
        xeog = self.conv_eog(x[1])
        xeeg_inshape = xeeg.shape
        xeog_inshape = xeog.shape
        xeeg = self.pos(xeeg.flatten(start_dim=2).permute(0,2,1)).permute(0,2,1).view(xeeg_inshape)
        xeog = self.pos(xeog.flatten(start_dim=2).permute(0,2,1)).permute(0,2,1).view(xeog_inshape)

        # print(x.shape)
        xeeg = self.att1_eeg(xeeg)
        xeog = self.att1_eog(xeog)

        eeg_token = self.avg_1(xeeg)
        eog_token = self.avg_1(xeog)

        xeeg = self.interm_eeg_conv_0(xeeg)
        xeog = self.interm_eog_conv_0(xeog)

        xeeg = torch.cat([xeeg, eog_token], dim=3)
        xeog = torch.cat([xeog, eeg_token], dim=3)

        xeeg = self.att2_eeg(xeeg)
        xeog = self.att2_eog(xeog)

        eeg_token = self.avg_2(xeeg[:,:,:,:-1])
        eog_token = self.avg_2(xeog[:,:,:,:-1])

        eeg_p_token = xeeg[:,:,:,-1].unsqueeze(dim=3)
        eog_p_token = xeog[:,:,:,-1].unsqueeze(dim=3)

        xeeg = self.interm_eeg_conv_1(xeeg)
        xeog = self.interm_eog_conv_1(xeog)

        xeeg = torch.cat([xeeg, eog_p_token, eog_token, self.cls_token.repeat(xeeg_inshape[0],1,1,1)], dim=3)
        xeog = torch.cat([xeog, eeg_p_token, eeg_token, self.cls_token.repeat(xeeg_inshape[0],1,1,1)], dim=3)

        xeeg = self.att3_eeg(xeeg)
        xeog = self.att3_eog(xeog)

        xm_eeg = self.avg_3(xeeg[:,:,:,:-3])
        xm_eog = self.avg_3(xeog[:,:,:,:-3])

        multi1_eeg = xeeg[:,:,:,-3].unsqueeze(dim=3)
        multi2_eeg = xeeg[:,:,:,-2].unsqueeze(dim=3)
        cls_eeg = xeeg[:,:,:,-1].unsqueeze(dim=3)

        multi1_eog = xeog[:,:,:,-3].unsqueeze(dim=3)
        multi2_eog = xeog[:,:,:,-2].unsqueeze(dim=3)
        cls_eog = xeog[:,:,:,-1].unsqueeze(dim=3)

        out = torch.cat([xm_eeg, multi1_eeg, multi2_eeg, cls_eeg, xm_eog, multi1_eog, multi2_eog, cls_eog],dim=1)

        return out

class EEG_Encoder_Single(nn.Module):
    def __init__(self, dec, transformer_type):
        super().__init__()

        # self.maxpool = nn.MaxPool2d(kernel_size=(1, 4))
        # self.pos1 = PositionalEncoder(d_model=dmodel*8, same_time_step=7)

        dmodel = 32 * dec

        self.conv = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, dmodel, kernel_size=(2, 5), stride=(1, 2), bias=False),
            # nn.ReLU(),
        )

        self.interm_conv_0 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(dmodel, dmodel, kernel_size=(1, 5), stride=(1, 2), bias=False),
            # nn.ReLU(),
        )
        self.interm_conv_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 2, 0, 0)),
            nn.Conv2d(dmodel, dmodel, kernel_size=(1, 5), stride=(1, 2), bias=False),
            # nn.ReLU(),
        )

        # self.latent = nn.Parameter(torch.randn(16, 8, dmodel))
        # self.cls_token = nn.Parameter(torch.randn(1, dmodel,1, 1))
        # self.cls_token = torch.cat([self.cls_token]*22, dim=0)
        self.pos = PositionalEncoder(d_model=dmodel, same_time_step=8)
        transformer_type = globals()[transformer_type]

        self.att1 = transformer_type(dmodel, 1)
        self.att2 = transformer_type(dmodel, 1)
        self.att3 = transformer_type(dmodel, 1)
        self.att4 = transformer_type(64*5, 1)
        self.att5 = transformer_type(64*5, 1)
        self.att6 = transformer_type(64*5, 1)

        self.avg = nn.AvgPool2d(kernel_size=(1, 75))


    def forward(self,x):
        x_shape = x[0].shape
        if len(x_shape)>4:
            for i in range(len(x)):
                x[i] = x[i].flatten(start_dim=0, end_dim=1)

        x = self.conv(x[0])
        x_inshape = x.shape
        x = self.pos(x.flatten(start_dim=2).permute(0,2,1)).permute(0,2,1).view(x_inshape)

        # print(x.shape)
        x = self.att1(x)
        x = self.interm_conv_0(x)
        x = self.att2(x)
        x = self.interm_conv_1(x)
        x = self.att3(x)
        x = self.avg(x)
        x = x.view(x_shape[0],x.shape[1]*x.shape[2]*x.shape[3], 1, x_shape[1])
        x = self.att4(x)
        x = self.att5(x)
        x = self.att6(x).permute(0,3,2,1)
        x = x.flatten(start_dim=0, end_dim = 1)

        return x

class EEG_Encoder_Ch_all(nn.Module):
    def __init__(self, dec, non_use):
        super().__init__()
        self.pad_1 = nn.ReflectionPad1d(2)
        self.conv1 = nn.Conv1d(1, 64*dec, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(64*dec, 128*dec, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(128*dec, 128*dec, kernel_size=3, stride=1)
        self.conv4 = nn.Conv1d(128*dec, 64*dec, kernel_size=1, stride=1)
        self.conv5 = nn.Conv1d(64*dec, 32*dec, kernel_size=3, stride=1)
        self.conv6 = nn.Conv1d(32*dec, 16*dec, kernel_size=1, stride=1)

        self.pad_2 = nn.ReflectionPad1d(1)
        # self.pad_2 = nn.ZeroPad2d((2,2,0,0))
        # self.conv2 = nn.Conv1d(10*dec, 20*dec, kernel_size=5, stride=1)

        self.maxpool_time = nn.MaxPool1d(4)
        self.avg_pool = nn.AvgPool1d(14)
        self.relu = torch.nn.ReLU()
        self.conv1_bn = nn.BatchNorm1d(1)
        self.conv2_bn = nn.BatchNorm1d(10*dec)

        # self.alpha = nn.Parameter(torch.randn(10*dec, 4),requires_grad=True)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.relu(self.conv1(self.pad_1(x)))
        x = self.relu(self.conv2(x))
        x = self.maxpool_time(x)

        x= self.relu(self.conv3(self.pad_2(x)))
        x = self.relu(self.conv4(x))

        x = self.maxpool_time(x)
        x = self.relu(self.conv5(self.pad_2(x)))
        x = self.relu(self.conv6(x))
        x = self.avg_pool(x)
        # x = self.maxpool_time(x)

        return x


class EEG_Shuffle_channels(nn.Module):
    def __init__(self, dec):
        super().__init__()

        # self.maxpool = nn.MaxPool2d(kernel_size=(1, 4))
        # self.pos1 = PositionalEncoder(d_model=dmodel*8, same_time_step=7)

        dmodel = 64 * dec

        self.conv = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 64 * dec, kernel_size=(1, 5), stride=(1,2), bias=False),

            # nn.ReflectionPad2d((2, 2, 1, 0)),
            # nn.Conv2d(64 * dec, 128 * dec, kernel_size=(1, 5), stride=(1, 2)),
            # nn.ReLU(),
            # # nn.MaxPool2d(kernel_size=(1, 2)),
            # nn.ReflectionPad2d((2, 2, 0, 0)),
            # nn.Conv2d(128 * dec, 16 * dec, kernel_size=(1, 5)),
            # nn.ReLU(),
        )

        # self.interm_conv_0 = nn.Sequential(
        #     nn.ReflectionPad2d((2, 2, 1, 1)),
        #     nn.Conv2d(64 * dec, 128 * dec, kernel_size=(2, 5), stride=(1,2)),
        #     # nn.ReLU(),
        # )
        # self.interm_conv_1 = nn.Sequential(
        #     nn.ReflectionPad2d((2, 2, 0, 0)),
        #     nn.Conv2d(128 * dec, 16 * dec, kernel_size=(2, 5)),
        #     # nn.ReLU(),
        # )
        # self.latent = nn.Parameter(torch.randn(16, 8, dmodel))
        # self.cls_token = nn.Parameter(torch.randn(1, 8, dmodel))
        # self.pos = PositionalEncoder(d_model=dmodel*8)

        self.att1 = My_Transformer_Layer_Ch_SmallFF(dmodel)
        # self.att1_1 = My_Transformer_Layer(dmodel*450)
        self.att2 = My_Transformer_Layer_Ch_SmallFF(dmodel*2)
        # self.att2_1 = My_Transformer_Layer(dmodel*2*225)
        self.att3 = My_Transformer_Layer_Ch_SmallFF(16 * dec)
        # self.att3_1 = My_Transformer_Layer(16 * dec*112)
        self.avg = nn.AvgPool2d(kernel_size=(1, 56))

        import random
        self.rands = random.sample(range(8), 8)
        # self.rands = [7,0,1,2,3,4,5,6,7,1]
        print("Our random shuffle is:")
        print(self.rands)
    def _shuffle_channels(self,x):
        return x[:,:,self.rands,:]

    def cross_attention(self,src):
        print(self.latent_space.shape)
        print(src.shape)
        src2 = self.self_attn(self.latent_space, self.latent_space, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def s_att(self,src):
        src2 = self.s_self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.s_norm1(src)
        src2 = self.s_linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.s_norm2(src)
        return src

    def forward(self, x):

        x = self.conv(x)
        x_shape = x.shape

        # x_shape = x.shape
        # x = x.permute(0,3,2,1)
        # mm= []
        # for i in range(8):
        #     m = []
        #     for j in range(8):
        #         if i!=j:
        #             tgt = self.multihead_cross_attn(x[:, :, i], x[:, :, j], x[:, :, j])[0]
        #             tgt = tgt + self.dropout2(x[:, :, i])
        #             tgt = self.norm2(tgt)
        #             tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        #             tgt = tgt + self.dropout3(tgt2)
        #             tgt = self.norm3(tgt)
        #             m.append(tgt.unsqueeze(dim=2))
        #     mm.append(torch.cat(m,dim=2).unsqueeze(dim=3))
        # x = torch.cat(mm,dim=3)
        # mm = []
        # for i in range(8):
        #     mm.append(self.attention1(self.pos1(x[:,:,:,i].flatten(start_dim=2))).unsqueeze(dim=2))
        # x = torch.cat(mm,dim=2)


        # m = self.dec_attention1(x[:,:,0],x[:,:,1])
        # x = self.perceiver(x)
        # print(x.shape)
        # m = []
        # for i in range(8):
        #     xi = x[:,:,i].permute(0,2,1)
        # x = x.view([x_shape[0],x_shape[3], x_shape[1] * x_shape[2]])
        # xi = x.clone()
        # l = torch.cat(x_shape[0]*[self.latent.unsqueeze(dim=0)],dim=0)
        # for i in range(1):
            # print(xi[:,:,i].permute(0,2,1).shape)
            # print(l.shape)

        # x = self.pos(x.flatten(start_dim=2)).view(x_shape)
        x = self.att1(x)#.permute(0,2,1,3)
        # x = self.att1_1(x).permute(0,2,1,3).view(x_shape)

        # x = self.interm_conv_0(x.permute(0,3,2,1)).permute(0,3,2,1)
        # x_shape = x.shape
        x = self.att2(x)#.permute(0,2,1,3)
        # x = self.att2_1(x).permute(0,2,1,3).view(x_shape)

        # x = self.interm_conv_1(x.permute(0,3,2,1)).permute(0,3,2,1)
        # x_shape = x.shape
        x = self.att3(x)#.permute(0,2,1,3)
        # x = self.att3_1(x).permute(0,2,1,3).view(x_shape)

        # x_shape = x.shape
        # x = self.pos(x.flatten(start_dim=2)).view(x_shape)
        # x = self.att1(x)
        # x = self.att2(x.flatten(start_dim=2)).permute(0,2,1)

        # x = self.attention2(x)
        # x = self.norm2(x)
        # x = self.ff2(x)
        # x = self.normff2(x)

            # x = self.attention3(x)
            # x = self.norm3(x)
            # x = self.ff3(x)
            # x = self.normff3(x)
        # print(x.shape)
        # m.append(k.unsqueeze(dim=2))
        # x = x.view([x_shape[0],16,8])
            # torch.cat(m,dim=2).permute(0,3,2,1)
        # x = x.permute(0,3,2,1).squeeze()
        # self.latent_space = nn.Parameter(torch.cat(x.shape[0] * [self.latent_space.unsqueeze(dim=0)],dim=0))
        #
        # for i in range(8):
        #     self.latent_space = self.cross_attention(x.squeeze())
        #     self.latent_space = self.s_att(self.latent_space)
        # print(self.latent_space.shape)
        # x = x.permute(0,3,1,2).flatten(start_dim=2, end_dim=3)
        # x = self.attention1(x)
        # # x = x.view(x_shape)
        # # x = self.maxpool(x)
        # # x_shape = x.shape
        # #
        # # # x = self.pos2(x)
        # # x = self.self_attention2(x.permute(0,3,1,2).flatten(start_dim=2))
        x = self.avg(x)

        # print(x.shape)
        # x = self._shuffle_channels(x)
        return x

class EEG_Transformer_SEDF(nn.Module):
    def __init__(self, dec, transformer_type):
        super().__init__()

        # self.maxpool = nn.MaxPool2d(kernel_size=(1, 4))
        # self.pos1 = PositionalEncoder(d_model=dmodel*8, same_time_step=7)

        dmodel = 32 * dec

        self.conv_eeg = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, dmodel, kernel_size=(2, 5), stride=(1,2), bias=False),
            # nn.ReLU(),
        )
        self.conv_eog = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, dmodel, kernel_size=(1, 5), stride=(1,2), bias=False),
            # nn.ReLU(),
        )

        self.interm_conv_0 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(dmodel, dmodel*2, kernel_size=(1, 5), stride=(1,2), bias=False),
            # nn.ReLU(),
        )
        self.interm_conv_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 2, 0, 0)),
            nn.Conv2d(dmodel*2, 32, kernel_size=(1, 5), stride=(1,2), bias=False),
            # nn.ReLU(),
        )
        # self.latent = nn.Parameter(torch.randn(16, 8, dmodel))
        # self.cls_token = nn.Parameter(torch.randn(1, 8, dmodel))
        self.pos = PositionalEncoder(d_model=dmodel, same_time_step=8)
        transformer_type = globals()[transformer_type]

        self.att1 = transformer_type(dmodel, 2)
        # self.att1_1 = My_Transformer_Layer(dmodel*450)
        self.att2 = transformer_type(dmodel*2, 2)
        # self.att2_1 = My_Transformer_Layer(dmodel*2*225)
        self.att3 = transformer_type(32, 2)
        self.att4 = transformer_type(1024, 1)
        self.att5 = transformer_type(1024, 1)
        self.att6 = transformer_type(1024, 1)
        # self.att3_1 = My_Transformer_Layer(16 * dec*112)
        self.avg = nn.AvgPool2d(kernel_size=(1, 23))

    def forward(self, x):
        x_shape = x[0].shape
        flag_seqtoseq = False
        if len(x_shape)>4:
            flag_seqtoseq = True
            for i in range(len(x)):
                x[i] = x[i].flatten(start_dim=0, end_dim=1)
        xeeg = self.conv_eeg(x[0])
        xeog = self.conv_eog(x[1])
        x = torch.cat([xeeg, xeog], dim=2)
        x_inshape = x.shape
        x = self.pos(x.flatten(start_dim=2).permute(0,2,1)).permute(0,2,1).view(x_inshape)
        # print(x.shape)
        x = self.att1(x)
        # x = self.att1_1(x)

        x = self.interm_conv_0(x)
        # x_shape = x.shape

        x = self.att2(x)
        # x = self.att2_1(x)

        x = self.interm_conv_1(x)
        # x_shape = x.shape
        x = self.att3(x)
        # x = self.att3_1(x)
        x = self.avg(x)

        if flag_seqtoseq:
            x = x.view(x_shape[0],x_shape[1],1,-1).permute(0,3,2,1)
            x = self.att4(x)
            x = self.att5(x)
            x = self.att6(x)
            x = x.permute(0,3,2,1).flatten(start_dim=0,end_dim=1)
        return x


class EEG_Transformer(nn.Module):
    def __init__(self, dec, transformer_type):
        super().__init__()

        # self.maxpool = nn.MaxPool2d(kernel_size=(1, 4))
        # self.pos1 = PositionalEncoder(d_model=dmodel*8, same_time_step=7)

        dmodel = 32 * dec

        self.conv = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, dmodel, kernel_size=(1, 5), stride=(1,2), bias=False),
            # nn.ReLU(),
        )

        self.interm_conv_0 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 1, 1)),
            nn.Conv2d(dmodel, dmodel*2, kernel_size=(2, 5), stride=(1,2), bias=False),
            # nn.ReLU(),
        )
        self.interm_conv_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 2, 0, 0)),
            nn.Conv2d(dmodel*2, 32, kernel_size=(2, 5), stride=(1,2), bias=False),
            # nn.ReLU(),
        )
        # self.latent = nn.Parameter(torch.randn(16, 8, dmodel))
        # self.cls_token = nn.Parameter(torch.randn(1, 8, dmodel))
        self.pos = PositionalEncoder(d_model=dmodel, same_time_step=8)
        transformer_type = globals()[transformer_type]

        self.att1 = transformer_type(dmodel)
        # self.att1_1 = My_Transformer_Layer(dmodel*450)
        self.att2 = transformer_type(dmodel*2, 3)
        # self.att2_1 = My_Transformer_Layer(dmodel*2*225)
        self.att3 = transformer_type(32)
        # self.att3_1 = My_Transformer_Layer(16 * dec*112)
        self.avg = nn.AvgPool2d(kernel_size=(1, 23))

    def forward(self, x):

        x = self.conv(x)
        x_shape = x.shape
        x = self.pos(x.flatten(start_dim=2).permute(0,2,1)).permute(0,2,1).view(x_shape)
        # print(x.shape)
        x = self.att1(x)
        # x = self.att1_1(x)

        x = self.interm_conv_0(x)
        # x_shape = x.shape

        x = self.att2(x)
        # x = self.att2_1(x)

        x = self.interm_conv_1(x)
        # x_shape = x.shape
        x = self.att3(x)
        # x = self.att3_1(x)
        x = self.avg(x)

        return x

class EEG_Transformer1(nn.Module):
    def __init__(self, dec, transformer_type):
        super().__init__()

        # self.maxpool = nn.MaxPool2d(kernel_size=(1, 4))
        # self.pos1 = PositionalEncoder(d_model=dmodel*8, same_time_step=7)

        dmodel = 32 * dec

        self.conv = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, dmodel, kernel_size=(1, 5), stride=(1,2), bias=False),
            # nn.ReLU(),
        )

        # self.interm_conv_0 = nn.Sequential(
        #     nn.ReflectionPad2d((2, 2, 1, 1)),
        #     nn.Conv2d(dmodel, dmodel*2, kernel_size=(2, 5), stride=(1,2), bias=False),
        #     # nn.ReLU(),
        # )
        # self.interm_conv_1 = nn.Sequential(
        #     nn.ReflectionPad2d((1, 2, 0, 0)),
        #     nn.Conv2d(dmodel*2, 32, kernel_size=(2, 5), stride=(1,2), bias=False),
        #     # nn.ReLU(),
        # )
        # self.latent = nn.Parameter(torch.randn(16, 8, dmodel))
        # self.cls_token = nn.Parameter(torch.randn(1, 8, dmodel))
        self.pos = PositionalEncoder(d_model=dmodel, same_time_step=8)
        transformer_type = globals()[transformer_type]

        self.att1 = transformer_type(dmodel)
        # self.att1_1 = My_Transformer_Layer(dmodel*450)
        self.att2 = transformer_type(dmodel)
        # self.att2_1 = My_Transformer_Layer(dmodel*2*225)
        self.att3 = transformer_type(dmodel)
        # self.att3_1 = My_Transformer_Layer(16 * dec*112)
        self.avg = nn.AvgPool2d(kernel_size=(1, 28*8))

    def forward(self, x):

        x = self.conv(x)
        x_shape = x.shape
        x = self.pos(x.flatten(start_dim=2).permute(0,2,1)).permute(0,2,1).view(x_shape)

        x = self.att1(x)
        # x = self.att1_1(x)

        # x = self.interm_conv_0(x)
        # x_shape = x.shape
        x = self.att2(x)
        # x = self.att2_1(x)

        # x = self.interm_conv_1(x)
        # x_shape = x.shape
        x = self.att3(x)
        # x = self.att3_1(x)

        x = self.avg(x)

        return x

class EEG_Transformer2(nn.Module):
    def __init__(self, dec, transformer_type):
        super().__init__()

        # self.maxpool = nn.MaxPool2d(kernel_size=(1, 4))
        # self.pos1 = PositionalEncoder(d_model=dmodel*8, same_time_step=7)

        dmodel = 32 * dec

        self.conv = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, dmodel, kernel_size=(1, 5), stride=(1,2), bias=False),
            # nn.ReLU(),
        )

        self.interm_conv_0 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(dmodel, 32, kernel_size=(1, 5), stride=(1,2), bias=False),
            # nn.ReLU(),
        )
        # self.interm_conv_1 = nn.Sequential(
        #     nn.ReflectionPad2d((1, 2, 0, 0)),
        #     nn.Conv2d(dmodel*2, 32, kernel_size=(2, 5), stride=(1,2), bias=False),
        #     # nn.ReLU(),
        # )
        # self.latent = nn.Parameter(torch.randn(16, 8, dmodel))
        # self.cls_token = nn.Parameter(torch.randn(1, 8, dmodel))
        self.pos = PositionalEncoder(d_model=dmodel, same_time_step=8)
        transformer_type = globals()[transformer_type]

        self.att1 = transformer_type(dmodel)
        # self.att1_1 = My_Transformer_Layer(dmodel*450)
        self.att2 = transformer_type(32)
        # self.att2_1 = My_Transformer_Layer(dmodel*2*225)
        self.att3 = transformer_type(32)
        # self.att3_1 = My_Transformer_Layer(16 * dec*112)
        self.avg = nn.AvgPool2d(kernel_size=(1, 28*2))

    def forward(self, x):

        x = self.conv(x)
        x_shape = x.shape
        x = self.pos(x.flatten(start_dim=2).permute(0,2,1)).permute(0,2,1).view(x_shape)

        x = self.att1(x)
        # x = self.att1_1(x)

        x = self.interm_conv_0(x)

        # x_shape = x.shape
        x = self.att2(x)
        # x = self.att2_1(x)

        # x = self.interm_conv_1(x)
        # x_shape = x.shape
        x = self.att3(x)
        # x = self.att3_1(x)

        x = self.avg(x)

        return x

class EEG_Transformer3(nn.Module):
    def __init__(self, dec, transformer_type):
        super().__init__()

        # self.maxpool = nn.MaxPool2d(kernel_size=(1, 4))
        # self.pos1 = PositionalEncoder(d_model=dmodel*8, same_time_step=7)

        dmodel = 32 * dec

        self.conv = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, dmodel, kernel_size=(1, 5), stride=(1,2), bias=False),
            # nn.ReLU(),
        )

        self.interm_conv_0 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(dmodel, dmodel*2, kernel_size=(1, 5), stride=(1,2), bias=False),
            # nn.ReLU(),
        )
        self.interm_conv_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 2, 0, 0)),
            nn.Conv2d(dmodel*2, 32, kernel_size=(1, 5), stride=(1,2), bias=False),
            # nn.ReLU(),
        )
        # self.latent = nn.Parameter(torch.randn(16, 8, dmodel))
        # self.cls_token = nn.Parameter(torch.randn(1, 8, dmodel))
        self.pos = PositionalEncoder(d_model=dmodel, same_time_step=8)
        transformer_type = globals()[transformer_type]

        self.att1 = transformer_type(dmodel)
        # self.att1_1 = My_Transformer_Layer(dmodel*450)
        self.att2 = transformer_type(dmodel*2)
        # self.att2_1 = My_Transformer_Layer(dmodel*2*225)
        self.att3 = transformer_type(32)
        # self.att3_1 = My_Transformer_Layer(16 * dec*112)
        self.avg = nn.AvgPool2d(kernel_size=(1, 28))

    def forward(self, x):

        x = self.conv(x)
        x_shape = x.shape
        x = self.pos(x.flatten(start_dim=2).permute(0,2,1)).permute(0,2,1).view(x_shape)

        x = self.att1(x)
        # x = self.att1_1(x)

        x = self.interm_conv_0(x)

        # x_shape = x.shape
        x = self.att2(x)
        # x = self.att2_1(x)

        x = self.interm_conv_1(x)

        # x_shape = x.shape
        x = self.att3(x)
        # x = self.att3_1(x)

        x = self.avg(x)

        return x

class EEG_Encoder_best_2d(nn.Module):
    def __init__(self, dec, _):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 64 * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 1, 1)),
            nn.Conv2d(64 * dec, 128 * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(128 * dec, 16 * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((1, 46)) #56
        )
        # import random
        # self.rands = random.sample(range(8), 8)

    def forward(self, x):

        # temp1 = copy.deepcopy(x[:,:,1,:])
        # temp3 = copy.deepcopy(x[:,:,3,:])
        # x[:, :, 1, :] = x[:,:,4,:]
        # x[:, :, 3, :] = x[:,:,6,:]
        # x[:, :, 4, :] = temp1
        # x[:, :, 6, :] = temp3
        return self.conv(x)

class EEG_Encoder_best_SEDF_1(nn.Module):
    def __init__(self, dec, _):
        super().__init__()
        size = 64
        self.conv_0_eeg = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, size * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_0_eog = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, size * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_1_eeg = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_1_eog = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_2_eeg = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d( size * dec, size * dec, kernel_size=(3, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_2_eog = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d( size * dec, size * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_0 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d( 1, 1, kernel_size=(3, 5), stride=(1, 1)),
            nn.ReLU(),
        )
        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(3, 5), stride=(1, 1)),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(3, 5), stride=(1, 1)),
            nn.ReLU(),
        )
        self.conv_3 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
        )
        self.avg = nn.AvgPool2d((1,187))

    def forward(self, x):

        x_shape = x[0].shape
        flag_seqtoseq = False
        if len(x_shape)>4:
            flag_seqtoseq = True
            for i in range(len(x)):
                x[i] = x[i].flatten(start_dim=0, end_dim=1)

        xeeg = x[0]
        xeog = x[1]

        x = self.conv_0(torch.cat([xeeg, xeog], dim=2))

        xeeg = self.conv_0_eeg(torch.cat([xeeg[:,:,0,:].unsqueeze(dim=2), x, xeeg[:,:,1,:].unsqueeze(dim=2)], dim=2))
        xeog = self.conv_0_eog(torch.cat([xeog, x], dim=2))

        x = self.conv_1(torch.cat([xeeg, xeog], dim=2))

        xeeg = self.conv_1_eeg(torch.cat([xeeg[:,:,0,:].unsqueeze(dim=2), x, xeeg[:,:,1,:].unsqueeze(dim=2)], dim=2))
        xeog = self.conv_1_eog(torch.cat([xeog, x], dim=2))

        x = self.conv_2(torch.cat([xeeg, xeog], dim=2))

        xeeg = self.conv_2_eeg(torch.cat([xeeg[:, :, 0, :].unsqueeze(dim=2), x, xeeg[:, :, 1, :].unsqueeze(dim=2)], dim=2))
        xeog = self.conv_2_eog(torch.cat([xeog, x], dim=2))

        x = self.conv_3(torch.cat([xeeg, xeog], dim=2))

        x = torch.cat([xeeg,x,xeog],dim=2)
        x = self.avg(x)

        if flag_seqtoseq:
            x = x.view([x_shape[0], x_shape[1], -1])
        else:
            x = x.view([x_shape[0],-1])

        return x

class EEG_Encoder_best_SEDF_2(nn.Module):
    def __init__(self, dec, _):
        super().__init__()
        size = 64
        self.conv_0_eeg = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, size * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_0_eog = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, size * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_1_eeg = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_1_eog = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_2_eeg = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d( size * dec, size * dec, kernel_size=(3, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_2_eog = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d( size * dec, size * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(3, 5), stride=(1, 1)),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(3, 5), stride=(1, 1)),
            nn.ReLU(),
        )
        self.conv_3 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
        )
        self.avg = nn.AvgPool2d((1,187))

    def forward(self, x):

        x_shape = x[0].shape
        flag_seqtoseq = False
        if len(x_shape)>4:
            flag_seqtoseq = True
            for i in range(len(x)):
                x[i] = x[i].flatten(start_dim=0, end_dim=1)


        xeeg = x[0]
        xeog = x[1]

        xeeg = self.conv_0_eeg(xeeg)
        xeog = self.conv_0_eog(xeog)

        x = self.conv_1(torch.cat([xeeg, xeog], dim=2))

        xeeg = self.conv_1_eeg(torch.cat([xeeg[:,:,0,:].unsqueeze(dim=2), x, xeeg[:,:,1,:].unsqueeze(dim=2)], dim=2))
        xeog = self.conv_1_eog(torch.cat([xeog, x], dim=2))

        x = self.conv_2(torch.cat([xeeg, xeog], dim=2))

        xeeg = self.conv_2_eeg(torch.cat([xeeg[:, :, 0, :].unsqueeze(dim=2), x, xeeg[:, :, 1, :].unsqueeze(dim=2)], dim=2))
        xeog = self.conv_2_eog(torch.cat([xeog, x], dim=2))

        x = self.conv_3(torch.cat([xeeg, xeog], dim=2))

        x = torch.cat([xeeg,x,xeog],dim=2)

        x = self.avg(x)

        if flag_seqtoseq:
            x = x.view([x_shape[0], x_shape[1], -1])
        else:
            x = x.view([x_shape[0],-1])

        return x

class EEG_TransferTransformer(nn.Module):
    def __init__(self, dec, transformer_type):
        super().__init__()

        dmodel = 128
        # # self.latent = nn.Parameter(torch.randn(16, 8, dmodel))
        # # self.cls_token = nn.Parameter(torch.randn(1, 8, dmodel))
        # self.pos_inner = PositionalEncoder(d_model=dmodel, same_time_step=1)
        # self.pos_inner_mod = PositionalEncoder(d_model=dmodel, same_time_step=1)
        # # transformer_type = globals()[transformer_type]
        modalities = 1
        # enc = nn.TransformerEncoderLayer(dmodel, nhead=8)
        # self.inner_att = nn.TransformerEncoder(enc,3)
        #
        # enc_mod = nn.TransformerEncoderLayer(dmodel, nhead=8)
        # self.inner_mod_att = nn.TransformerEncoder(enc_mod,3)
        # # self.inner_att = nn.GRU(dmodel,dmodel,num_layers=3)
        #
        # self.pos_outer = PositionalEncoder(d_model=dmodel*modalities, same_time_step=1)
        # enc_outer = nn.TransformerEncoderLayer(dmodel*modalities, nhead=8)
        # self.outer_att = nn.TransformerEncoder(enc_outer,3)
        # # self.outer_att = nn.GRU(dmodel*modalities,dmodel*modalities,num_layers=3)
        # self.avg = nn.AvgPool2d((1,20))
        # self.att = Attention(dmodel)
        # self.mod_att = Attention(dmodel)

        self.tf1 = Multi_Transformer(dmodel, inner= 20, outer = 21, modalities=8, heads=8,
                                     layers = [ "fourier_pos", "inner_mod_att","aggregation_att_contx_inner", "fourier_pos", "outer_att"], num_layers=4, pos = False)

        # self.tf2 = Multi_Transformer(dmodel,modalities=1, heads=23, layers=3, pos = True)
        # self.tf3 = Multi_Transformer(dmodel,modalities=1, heads=23, layers=3, pos = True)

    def forward(self, x):
        # x = torch.einsum("ijkmn->ijmkn", x)
        x_shape = x.shape
        b, outer, inner, modalities = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        # print(x.shape)
        x = self.tf1(x)
        # x = self.tf2(x)
        # x = self.tf3(x)

        # x = einops.rearrange(x,"b outer inner mod k->inner (b outer mod) k" )
        # w = self.att(x)
        # x = torch.einsum("ijk,jmi -> mjk", x, w)
        # x = einops.rearrange(x,"inner (b outer mod) k-> outer (b inner mod) k",b=b, outer=outer, mod=modalities)
        # x = self.outer_att(x)
        # x = einops.rearrange(x,"outer (b mod) k->b outer (mod k)",b=b, mod=modalities )

        # x = einops.rearrange(x,"b outer inner mod k->(outer inner mod) b k")
        # x = self.outer_att(x)
        # x = einops.rearrange(x,"(outer inner mod) b k->b outer (inner mod k)", mod=modalities, inner =inner, outer=outer )

        # x_shape = x.shape
        # if (len(x_shape)>3):
        #     x = x.flatten(start_dim=0,end_dim=2)
        # x = self.pos_inner(x).permute(1,0,2)
        # # x = x.permute(1,0,2)
        #
        # # xeog = self.conv_eog(xeog)
        # # x = torch.cat([xeeg,xeog],dim=2)
        # # x = self.pos(x.flatten(start_dim=2).permute(0, 2, 1)).permute(0, 2, 1).view(x_inner_shape)
        # print(x.shape)
        #
        # x = self.inner_att(x)
        # w = self.att(x)
        # x = torch.einsum("ijk,jmi -> mjk",x,w)
        # x = x.view([x_shape[0]*x_shape[1],x_shape[2], -1])
        #
        # x = self.pos_inner_mod(x).permute(1,0,2)
        #
        # x = self.inner_mod_att(x)
        # w = self.mod_att(x)
        # x = torch.einsum("ijk,jmi -> mjk",x,w)
        #
        # # print(x.shape)
        # x = x.permute(1,2,0)
        # # x = self.avg(x)# average pooling
        # x = x.view([x_shape[0], x_shape[1], -1])
        # x = self.pos_outer(x.permute(1,0,2)).permute(1,0,2)
        # # x = x.permute(1,0,2).permute(1,0,2)
        # x = self.outer_att(x)
        # # print(x.shape)
        # # x = self.avg(x).flatten(start_dim=1)
        # # x = x.view([x_shape[0],x_shape[1], -1]).permute(1,0,2)
        # # x = self.outer_att(x).permute(1,0,2).unsqueeze(dim=2)
        return x

class TF_inner_mod_att_diff_fc(nn.Module):
    def __init__(self, d_model, nhead, modalities=1, dim_feedforward=1024,  dropout=0.1, activation="relu"):
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


        self.mod_0_linear1 = nn.Linear(d_model, dim_feedforward)
        self.mod_0_dropout = nn.Dropout(dropout)
        self.mod_0_linear2 = nn.Linear(dim_feedforward, d_model)
        self.mod_0_norm2 = nn.LayerNorm(d_model)
        self.mod_0_dropout2 = nn.Dropout(dropout)

        self.mod_1_linear1 = nn.Linear(d_model, dim_feedforward)
        self.mod_1_dropout = nn.Dropout(dropout)
        self.mod_1_linear2 = nn.Linear(dim_feedforward, d_model)
        self.mod_1_norm2 = nn.LayerNorm(d_model)
        self.mod_1_dropout2 = nn.Dropout(dropout)

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

        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        src = einops.rearrange(src, "b outer inner mod k -> (inner mod) (b outer) k")

        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = einops.rearrange(src, "(inner mod) b_outer k -> mod inner b_outer k", inner = self.inner, mod = self.mod)

        src2 = self.mod_0_linear2(self.mod_0_dropout(self.activation(self.mod_0_linear1(src[0]))))
        src[0] = src[0] + self.mod_0_dropout2(src2)
        src[0] = self.mod_0_norm2(src[0])

        src2 = self.mod_1_linear2(self.mod_1_dropout(self.activation(self.mod_1_linear1(src[1]))))
        src[1] = src[0] + self.mod_1_dropout2(src2)
        src[1] = self.mod_1_norm2(src[1])

        src = einops.rearrange(src, "mod inner (b outer) k -> b outer inner mod k",b=self.batch, outer=self.outer, mod=self.mod)

        return src

class BertNormOutput(nn.Module):  # This class is added by Goro Kobayashi
    def __init__(self, num_attention_heads, hidden_size):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    def forward(self, hidden_states, attention_probs, value_layer, dense, LayerNorm, pre_ln_states):
        # Args:
        #   hidden_states: Representations from previous layer and inputs to self-attention. (batch, seq_length, all_head_size)
        #   attention_probs: Attention weights calculated in self-attention. (batch, num_heads, seq_length, seq_length)
        #   value_layer: Value vectors calculated in self-attention. (batch, num_heads, seq_length, head_size)
        #   dense: Dense layer in self-attention. nn.Linear(all_head_size, all_head_size)
        #   LayerNorm: nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        #   pre_ln_states: Vectors just before LayerNorm (batch, seq_length, all_head_size)

        with torch.no_grad():

            # Make transformed vectors f(x) from Value vectors (value_layer) and weight matrix (dense).
            dense = dense.weight.view(self.all_head_size, self.num_attention_heads, self.attention_head_size)
            transformed_layer = torch.einsum('bhsv,dhv->bhsd', value_layer, dense)

            # Make weighted vectors αf(x) from transformed vectors (transformed_layer)
            # and attention weights (attentions):
            # (batch, num_heads, seq_length, seq_length, all_head_size)
            weighted_layer = torch.einsum('bhks,bhsd->bhksd', attention_probs, transformed_layer)
            weighted_norm = torch.norm(weighted_layer, dim=-1)

            # Sum each weighted vectors αf(x) over all heads:
            # (batch, seq_length, seq_length, all_head_size)
            summed_weighted_layer = weighted_layer.sum(dim=1)
            summed_weighted_norm = torch.norm(summed_weighted_layer, dim=-1)

            """ここからがnew"""
            # Make residual matrix (batch, seq_length, seq_length, all_head_size)
            hidden_shape = hidden_states.size()  # (batch, seq_length, all_head_size)
            device = hidden_states.device
            residual = torch.einsum('sk,bsd->bskd', torch.eye(hidden_shape[1]).to(device), hidden_states)

            # Make matrix of summed weighted vector + residual vectors
            residual_weighted_layer = summed_weighted_layer + residual
            residual_weighted_norm = torch.norm(residual_weighted_layer, dim=-1)

            # consider layernorm
            ln_weight = LayerNorm.weight.data
            ln_eps = LayerNorm.eps

            # 実際にLayerNormにかけられるベクトル pre_ln_states の平均・分散を計算
            mean = pre_ln_states.mean(-1, keepdim=True)  # (batch, seq_len, 1)
            var = (pre_ln_states - mean).pow(2).mean(-1, keepdim=True).unsqueeze(dim=2)  # (batch, seq_len, 1, 1)

            # attention + residual のサムの中のベクトルごとに平均を計算
            each_mean = residual_weighted_layer.mean(-1, keepdim=True)  # (batch, seq_len, seq_len, 1)

            # attention + residual のサムの中の各ベクトルから，各平均を引き，標準偏差で割る
            # (LayerNorm の normalization 部分をサムの中のベクトルごとに実行していることに相当)
            normalized_layer = torch.div(residual_weighted_layer - each_mean,
                                         (var + ln_eps) ** (1 / 2))  # (batch, seq_len, seq_len, all_head_size)

            # さらに，LayerNorm の重みでエレメント積を各ベクトルに対して実行
            post_ln_layer = torch.einsum('bskd,d->bskd', normalized_layer,
                                         ln_weight)  # (batch, seq_len, seq_len, all_head_size)
            post_ln_norm = torch.norm(post_ln_layer, dim=-1)  # (batch, seq_len, seq_len)

            # Attn-N の mixing ratio
            attn_preserving = torch.diagonal(summed_weighted_layer, dim1=1, dim2=2).permute(0, 2, 1)
            attn_mixing = torch.sum(summed_weighted_layer, dim=2) - attn_preserving
            attn_preserving_norm = torch.norm(attn_preserving, dim=-1)
            attn_mixing_norm = torch.norm(attn_mixing, dim=-1)
            attn_n_mixing_ratio = attn_mixing_norm / (attn_mixing_norm + attn_preserving_norm)

            # AttnRes-N の mixing ratio
            before_ln_preserving = torch.diagonal(residual_weighted_layer, dim1=1, dim2=2).permute(0, 2, 1)
            before_ln_mixing = torch.sum(residual_weighted_layer, dim=2) - before_ln_preserving
            before_ln_preserving_norm = torch.norm(before_ln_preserving, dim=-1)
            before_ln_mixing_norm = torch.norm(before_ln_mixing, dim=-1)
            attnres_n_mixing_ratio = before_ln_mixing_norm / (before_ln_mixing_norm + before_ln_preserving_norm)

            # AttnResLn-N の mixing ratio
            post_ln_preserving = torch.diagonal(post_ln_layer, dim1=1, dim2=2).permute(0, 2, 1)
            post_ln_mixing = torch.sum(post_ln_layer, dim=2) - post_ln_preserving
            post_ln_preserving_norm = torch.norm(post_ln_preserving, dim=-1)
            post_ln_mixing_norm = torch.norm(post_ln_mixing, dim=-1)
            attnresln_n_mixing_ratio = post_ln_mixing_norm / (post_ln_mixing_norm + post_ln_preserving_norm)

            # kkontras
            # AttnResLn-N の mixing ratio of three neighbors in comparison with the rest
            post_ln_preserving = torch.diagonal(post_ln_layer, dim1=1, dim2=2).permute(0, 2, 1)
            post_ln_mixing = torch.sum(post_ln_layer, dim=2) - post_ln_preserving
            post_ln_preserving_norm = torch.norm(post_ln_preserving, dim=-1)
            post_ln_mixing_norm = torch.norm(post_ln_mixing, dim=-1)
            attnresln_n_mixing_ratio = post_ln_mixing_norm / (post_ln_mixing_norm + post_ln_preserving_norm)

            outputs = (weighted_norm,  # ||αf(x)||
                       summed_weighted_norm,  # ||Σαf(x)||
                       residual_weighted_norm,  # ||Σαf(x) + x||
                       post_ln_norm,  # Norm of vectors after LayerNorm
                       attn_n_mixing_ratio,  # Mixing ratio for Attn-N
                       attnres_n_mixing_ratio,  # Mixing ratio for AttnRes-N
                       attnresln_n_mixing_ratio,  # Mixing ratio for AttnResLn-N
                       )
        return outputs

class ScaledDotProductAttention(nn.Module):
    def __init__(self, rpos=False, d_head=16, max_len=7, head_num=8):
        super().__init__()
        self.rpos = rpos
        self.head_num = head_num
        if rpos:
            self.k_rpos = Relative_Positional_Embeddings(tokens=max_len, dim_head=d_head, heads=head_num)
            self.v_rpos = Relative_Positional_Embeddings(tokens=max_len, dim_head=d_head, heads=head_num)

    def forward(self, query, key, value, gbiased, prevalue, mask=None):
        query = einops.rearrange(query,"seq b f -> b seq f")
        key = einops.rearrange(key,"seq b f -> b seq f")
        value = einops.rearrange(value,"seq b f -> b seq f")
        # attn_output, att_weights = F._scaled_dot_product_attention(query, key, value, attn_mask=mask)
        dk = query.size()[-1]

        if self.rpos:
            rel_key = einops.rearrange(key,"(b h) seq f -> b h seq f", b = int(key.shape[0]/self.head_num), h = self.head_num)
            rel_key = self.k_rpos(rel_key)
            rel_key = einops.rearrange(rel_key, " b h seq f -> (b h) seq f ")
            scores = (query.matmul(key.transpose(-2, -1)) + rel_key)/ math.sqrt(dk)
        else:
            scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)

        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)

        if gbiased:
            attention = gbiased(scores, prevalue)
        else:
            attention = nn.functional.softmax(scores, dim=-1)

        attn_output = torch.einsum('b i j , b j d -> b i d', attention, value)
        attn_output = einops.rearrange(attn_output," b seq f -> seq b f")

        return attn_output, attention
class ScaledDotProductAttention_HP(nn.Module):
    def __init__(self, rpos=False, d_head=16, max_len=7, head_num=8):
        super().__init__()
        self.rpos = rpos
        self.head_num = head_num
        if rpos:
            self.k_rpos = Relative_Positional_Embeddings(tokens=max_len, dim_head=d_head, heads=head_num)
            self.v_rpos = Relative_Positional_Embeddings(tokens=max_len, dim_head=d_head, heads=head_num)

    def forward(self, query, key, value, gbiased, prevalue, mask=None):

        mod_shapes = [query[i].shape[0] for i in range(2)]
        q = torch.cat([einops.rearrange(query[i],"seq b f -> b seq f") for i in range(2)],dim=1)
        k = torch.cat([einops.rearrange(key[i],"seq b f -> b seq f") for i in range(2)],dim=1)
        v = torch.cat([einops.rearrange(value[i],"seq b f -> b seq f") for i in range(2)],dim=1)

        # attn_output, att_weights = F._scaled_dot_product_attention(query, key, value, attn_mask=mask)
        dk = q.size()[-1]


        if self.rpos:
            rel_key = einops.rearrange(k,"(b h) seq f -> b h seq f", b = int(k.shape[0]/self.head_num), h = self.head_num)
            rel_key = self.k_rpos(rel_key)
            rel_key = einops.rearrange(rel_key, " b h seq f -> (b h) seq f ")
            scores = (q.matmul(k.transpose(-2, -1)) + rel_key)/ math.sqrt(dk)
        else:
            scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(dk)

        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)

        if gbiased:
            attention = gbiased(scores, v)
        else:
            attention = nn.functional.softmax(scores, dim=-1)

        attn_output = torch.einsum('b i j , b j d -> b i d', attention, v)
        attn_output = einops.rearrange(attn_output," b seq f -> seq b f")

        return [attn_output[:mod_shapes[0]],attn_output[mod_shapes[0]:]], attention

class ScaledDotProductAttention_Sparse(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention_Sparse, self).__init__()
        self.c = nn.Parameter(torch.Tensor([10000]), requires_grad=False)
        self.step_function = nn.Parameter(torch.Tensor([0]), requires_grad=False)

        self.is_bidirectional = True
        self.stride = 6
        self.expressivity = 1

    def compute_fixed_attention_subset(self, word_index, tgt_len):
        # +1s account for range function; [min, max) -> [min, max]
        if not self.is_bidirectional:
            absolute_max = word_index + 1
        else:
            absolute_max = tgt_len

        # Subset 1 - whole window
        rounded_index = (
            math.floor((word_index + self.stride) / self.stride) * self.stride
        )
        if word_index % self.stride == 0 and word_index != 0:
            subset_one = set(
                range(word_index - self.stride, min(absolute_max, word_index + 1))
            )
        else:
            subset_one = set(
                range(
                    max(0, rounded_index - self.stride),
                    min(absolute_max, rounded_index + 1),
                )
            )

        # Subset 2 - summary per window
        # If bidirectional, subset 2 is the same for every index
        subset_two = set()
        if not self.is_bidirectional:
            subset_two = self.compute_subset_summaries(absolute_max)

        return subset_one.union(subset_two)

    # Computes Ai(2)
    def compute_subset_summaries(self, absolute_max):
        checkpoint_index = self.compute_checkpoint(0)
        subset_two = set()
        while checkpoint_index <= absolute_max - 1:
            print(checkpoint_index)
            summary = set(
                range(
                    checkpoint_index,
                    min(checkpoint_index + self.expressivity + 1, absolute_max),
                )
            )
            subset_two = subset_two.union(summary)
            checkpoint_index = self.compute_checkpoint(checkpoint_index + self.stride)
        return subset_two

    # Used for Ai(2) calculations - beginning of [l-c, l] range
    def compute_checkpoint(self, word_index):
        if word_index % self.stride == 0 and word_index != 0:
            checkpoint_index = word_index - self.expressivity
        else:
            checkpoint_index = (
                math.floor(word_index / self.stride) * self.stride
                + self.stride
                - self.expressivity
            )
        return checkpoint_index

    # Compute sparse mask - if bidirectional, can pre-compute and store
    def buffered_sparse_mask(self, tensor, tgt_len, src_len):
        assert tgt_len > self.stride
        sparse_mask = torch.empty((tgt_len, src_len)).float().fill_(float("-inf"))

        # # If bidirectional, subset 2 is the same for every index
        # subset_summaries = set()
        # if self.is_bidirectional:
        #     subset_summaries = self.compute_subset_summaries(tgt_len)

        for i in range(tgt_len):
            fixed_attention_subset = self.compute_fixed_attention_subset(i, tgt_len)
            # fixed_attention_subset = fixed_attention_subset.union(subset_summaries)
            included_word_indices = torch.LongTensor(list(fixed_attention_subset))
            sparse_mask[i].index_fill_(0, included_word_indices, 0)
        return sparse_mask.type_as(tensor)

    def apply_sparse_mask(self, attn_weights):
        print(attn_weights.shape)
        bsz, tgt_len, src_len = attn_weights.shape
        sparse_mask = self.buffered_sparse_mask(attn_weights, tgt_len, src_len)

        sparse_mask = sparse_mask.unsqueeze(0).expand(
            bsz, tgt_len, src_len
        )
        attn_weights += sparse_mask

    def forward(self, query, key, value, query_sparse, key_sparse, mask=None):
        query = einops.rearrange(query,"seq b f -> b seq f")
        query_sparse = einops.rearrange(query_sparse,"seq b f -> b seq f")
        key = einops.rearrange(key,"seq b f -> b seq f")
        key_sparse = einops.rearrange(key_sparse,"seq b f -> b seq f")
        value = einops.rearrange(value,"seq b f -> b seq f")
        # attn_output, att_weights = F._scaled_dot_product_attention(query, key, value, attn_mask=mask)
        dk = query.size()[-1]

        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        # M =  query_sparse.matmul(key_sparse.transpose(-2, -1))

        # scores = torch.exp(scores)
        # mean_value = scores.mean()
        # scores.data[scores<mean_value] = 0

        # sparsity_M = self.c * torch.heaviside(M, self.step_function)
        # #
        # scores -= sparsity_M

        import matplotlib.pyplot as plt
        # plt.imshow(copy.deepcopy(scores[0]).cpu().detach().numpy(), cmap='hot', interpolation='nearest')
        # plt.show()
        self.apply_sparse_mask(scores)

        attention = F.softmax(scores, dim=-1)


        attn_output = torch.einsum('b i j , b j d -> b i d', attention, value)

        attn_output = einops.rearrange(attn_output," b seq f -> seq b f")

        return attn_output, attention

#Attention Bias Choices
class Gaussian_Attention_Bias(nn.Module):

    def __init__(self, rate=1, std=0.3, temp=5, type="add", with_diagonal=False, n_position=200, rand_runs=500):
        super().__init__()
        self.rate = rate
        self.rand_runs = rand_runs
        self.n_positions = n_position
        self.type = type

        assert (type!="mul" or type != "add"  or type != "pass"), "Type of neigh bias should be 'mul' or 'add' or 'pass'"

        gaussian_grads_diag = torch.zeros([self.rand_runs, n_position, n_position])
        number_of_diagonals = n_position+1 if with_diagonal else n_position
        diag_scramble = Parallel(n_jobs=8)(delayed(self._parallel_get_diagonal)( i_diagonal, n_position, temp, std, gaussian_grads_diag) for i_diagonal in tqdm(range(number_of_diagonals)))
        self.gaussian_grads_diag = nn.Parameter(self._gather_diagonals(diag_scramble, torch.zeros([self.rand_runs, n_position, n_position])), requires_grad=False)
        self.softmax_inf_regulator = nn.Parameter(torch.FloatTensor([float("-inf")]), requires_grad=False)
        self.epsilon = nn.Parameter(torch.FloatTensor([0.00001]), requires_grad=False)

    def _parallel_get_diagonal(self, i_diagonal, n_position, temp, std, gaussian_grads_diag):
        diagonal_vals = torch.ones(i_diagonal, dtype=torch.long)
        diag_matrix = torch.diagflat(diagonal_vals, offset=n_position - i_diagonal) + torch.diagflat(diagonal_vals, offset=-n_position + i_diagonal)
        diag_matrix = diag_matrix.unsqueeze(0).repeat(self.rand_runs, 1, 1)
        mean = i_diagonal if temp == 0 else i_diagonal / temp
        gaussian_grads_diag[diag_matrix > 0] = torch.empty([self.rand_runs, n_position, n_position]).normal_(mean=mean, std=std)[diag_matrix > 0]
        return gaussian_grads_diag

    def _gather_diagonals(self, diag_scramble, gaussian_grads_diag):
        for i in diag_scramble:
            gaussian_grads_diag += i
        gaussian_grads_diag = nn.functional.softmax(nn.functional.softmax(gaussian_grads_diag, dim=-1), dim=-1)
        return gaussian_grads_diag

    def forward(self, attention_weigths, prevalues):

        seq = attention_weigths.shape[-1]
        rand_init_batch = torch.randint(0, self.rand_runs - attention_weigths.shape[0], (1,))
        rand_init =  torch.randint(0, self.n_positions - seq, (1,))
        attention_weigths = nn.functional.softmax(attention_weigths, dim=-1)

        if self.type == "add":
            attention_weigths = nn.functional.softmax(attention_weigths, dim=-1)
            biased_attention_weights = attention_weigths +  self.rate * self.gaussian_grads_diag[rand_init_batch:rand_init_batch+attention_weigths.shape[0],rand_init:rand_init+seq,rand_init:rand_init+seq]
            biased_attention_weights = nn.functional.log_softmax(biased_attention_weights, dim=-1)

        elif self.type == "mul":
            attention_weigths = nn.functional.softmax(attention_weigths, dim=-1)
            biased_attention_weights = attention_weigths * self.gaussian_grads_diag[rand_init_batch:rand_init_batch+attention_weigths.shape[0],rand_init:rand_init+seq,rand_init:rand_init+seq]
            biased_attention_weights = (biased_attention_weights - torch.min(biased_attention_weights, dim=-1)[0]) / (
                        torch.max(biased_attention_weights, dim=-1)[0] - torch.min(biased_attention_weights, dim=-1)[0] + self.epsilon)

        elif self.type == "pass":
            biased_attention_weights = self.gaussian_grads_diag[rand_init_batch:rand_init_batch+attention_weigths.shape[0],rand_init:rand_init+seq,rand_init:rand_init+seq]

        # biased_attention_weights[biased_attention_weights<0.001] = self.softmax_inf_regulator #This line helps to maintain zeros after softmax
        # biased_attention_weights = nn.functional.log_softmax(biased_attention_weights, dim=-1)
        # print("neigh")
        # print(attention_weigths[0][0])
        # print(self.gaussian_grads_diag[rand_init_batch:rand_init_batch+attention_weigths.shape[0],rand_init:rand_init+seq,rand_init:rand_init+seq][0][0])
        # print(biased_attention_weights[0][0])

        # plt.subplot(131)
        # plt.imshow(attention_weigths[0].detach().cpu().numpy(), cmap="Blues")
        # plt.axis("off")
        # plt.title("Attention Weights")
        # plt.subplot(132)
        # plt.imshow(self.gaussian_grads_diag[rand_init_batch:rand_init_batch+attention_weigths.shape[0],rand_init:rand_init+seq,rand_init:rand_init+seq][0].detach().cpu().numpy(), cmap="Blues")
        # plt.axis("off")
        # plt.title("Gaussian Bias")
        # plt.subplot(133)
        # plt.imshow(biased_attention_weights[0].detach().cpu().numpy(), cmap="Blues")
        # plt.axis("off")
        # plt.title("Add Output")
        # plt.show()

        return biased_attention_weights

class Gaussian_Learned_Attention_Bias(nn.Module):

    def __init__(self, dmodel, type="add" , rate=1, heads=8, with_diag=True):
        super().__init__()
        self.heads = heads
        self.rate = rate
        self.type = type
        self.with_diag = with_diag

        assert (type!="mul" or type != "add"  or type != "pass"), "Type of neigh bias should be 'mul' or 'add' or 'pass'"

        dh = int(dmodel/heads)
        # self.linear_pw = nn.Linear(dh, dh)
        # self.linear_pu = nn.Linear(dh, 1)
        self.linear_sw = nn.Linear(dh, dh)
        self.linear_su = nn.Linear(dh, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, attention_weigths, prevalues):

        seq = attention_weigths.shape[-1]
        prevalues = einops.rearrange(prevalues,"seq b (h f) -> (b h) seq f", h=self.heads)

        #If u want to calculate also the centers of the gaussians
        # pi = seq * self.sigmoid(self.linear_pu(self.tanh(self.linear_pw(prevalues))))
        #We use as centers the main diagonal.
        pi = torch.arange(0,seq, device=prevalues.device).unsqueeze(0).unsqueeze(2).repeat(prevalues.shape[0], 1, 1)

        si = seq * self.sigmoid(self.linear_su(self.tanh(self.linear_sw(prevalues))))


        en = - torch.pow((pi.squeeze().unsqueeze(1).repeat(1, seq, 1) - pi.repeat(1, 1, seq)), 2)
        den = 2 * torch.pow(si.repeat(1, 1, seq), 2)
        gaussian_bias = en / den
        # gaussian_bias = self.softmax( en / den )


        if not self.with_diag:
            diagonal_vals = torch.ones(seq, dtype=torch.long)
            diag_matrix = torch.diagflat(diagonal_vals)
            gaussian_bias[:, diag_matrix>0] *= torch.zeros([1], device=gaussian_bias.device)


        if self.type == "add":
            attention_weigths = nn.functional.softmax(attention_weigths, dim=-1)
            biased_attention_weights = attention_weigths +  self.rate * gaussian_bias
        elif self.type == "mul":
            attention_weigths = nn.functional.softmax(attention_weigths, dim=-1)
            biased_attention_weights = attention_weigths * gaussian_bias
        elif self.type == "pass":
            biased_attention_weights = gaussian_bias

        biased_attention_weights = nn.functional.softmax(biased_attention_weights, dim=-1)

        # print("neigh")
        # print(attention_weigths[0][0])
        # print(gaussian_bias[0][0])
        # print(biased_attention_weights[0][0])
        #
        # plt.subplot(131)
        # plt.imshow(attention_weigths[0].detach().cpu().numpy(), cmap="Blues")
        # plt.axis("off")
        # plt.title("Attention Weights")
        # plt.subplot(132)
        # plt.imshow(gaussian_bias[0].detach().cpu().numpy(), cmap="Blues")
        # plt.axis("off")
        # plt.title("Learned Gaussian Bias")
        # plt.subplot(133)
        # plt.imshow(biased_attention_weights[0].detach().cpu().numpy(), cmap="Blues")
        # plt.axis("off")
        # plt.title("Add Output")
        # plt.show()

        return biased_attention_weights

class Attention_Bias_Neigh(nn.Module):

    def __init__(self, rate=1, type="mul", num_diagonals=2, with_diagonal=False, n_position=200):
        """

        :param rate: (int) Rate of the bias in case we use type=="add"
        :param type: (string) 'mul'-> multiply extracted attention with neigh bias
                           or 'add'-> add extracted attention with neigh bias multiplied by rate
                           or 'pass'->discard extracted attention and return uniform neigh bias
        :param num_diagonals: (int) How many diagonals are considered neighborhood. Each one has two diagonals, symmetrical to the main diagonal
        :param with_diagonal: (bool) True = include the main diagonal in the neigh mask
        :param n_position: (int) Max seq length
        """
        super().__init__()
        self.rate = rate
        self.n_positions = n_position
        self.type = type

        assert (type!="mul" or type != "add"  or type != "pass"), "Type of neigh bias should be 'mul' or 'add' or 'pass'"

        i_diagonal = n_position-1
        diagonal_vals = torch.ones(i_diagonal, dtype=torch.long)
        diag_matrix = torch.diagflat(diagonal_vals, offset=n_position - i_diagonal) + torch.diagflat(diagonal_vals, offset=-n_position + i_diagonal)

        if with_diagonal:
            i_diagonal = n_position
            diagonal_vals = torch.ones(i_diagonal, dtype=torch.long)
            diag_matrix += torch.diagflat(diagonal_vals)

        if num_diagonals > 1:
            for i in range(2,num_diagonals+1):
                i_diagonal = n_position-i

                diagonal_vals = torch.ones(i_diagonal, dtype=torch.long)
                diag_matrix += torch.diagflat(diagonal_vals, offset=n_position - i_diagonal) + torch.diagflat(diagonal_vals, offset=-n_position + i_diagonal)

        self.neigh_diag = nn.Parameter(diag_matrix.float(), requires_grad=False)
        self.softmax_inf_regulator = nn.Parameter(torch.FloatTensor([float("-inf")]), requires_grad=False)

    def forward(self, attention_weigths, prevalues):
        """

        :param attention_weigths: Attention weights after softmax, [batch*h*(maybe inner seq), seq, seq]
        :param prevalues: Not used here, aims for other attention bias techniques.
        :return:
        """
        seq = attention_weigths.shape[-1]

        rand_init =  torch.randint(0, self.n_positions - seq, (1,))

        # attention_bias = self.neigh_diag.unsqueeze(0).repeat(attention_weigths.shape[0], 1, 1)

        if self.type == "mul":
            biased_attention_weights = attention_weigths * self.neigh_diag[rand_init:rand_init + seq, rand_init:rand_init + seq]
        elif self.type == "add":
            attention_weigths = nn.functional.softmax(attention_weigths, dim=-1)
            biased_attention_weights = attention_weigths + self.rate * nn.functional.softmax(self.neigh_diag[rand_init:rand_init + seq, rand_init:rand_init + seq], dim=-1)
        elif self.type == "pass":
            biased_attention_weights = self.neigh_diag[rand_init:rand_init + seq, rand_init:rand_init + seq]

        biased_attention_weights[biased_attention_weights==0] = self.softmax_inf_regulator #This line helps to maintain zeros after softmax
        biased_attention_weights = nn.functional.softmax(biased_attention_weights, dim=-1)

        return biased_attention_weights

class Multimodal_Dropout_outer(nn.Module):

    def __init__(self, dmodel, dropout_prob = 0.1):

        super().__init__()
        self.dropout_prob = dropout_prob
        mask = torch.rand([1,dmodel])
        self.masked_token = nn.Parameter(mask, requires_grad=False)


    def forward(self, input):

        input_shape = input.shape

        input = einops.rearrange(input, "b outer inner mod k -> (b outer inner mod) k")

        dropout_idxs = torch.Tensor([1-self.dropout_prob]).repeat((input.shape[0]))
        dropout_idxs = torch.bernoulli(dropout_idxs)
        input[dropout_idxs > 0] = self.masked_token.repeat((int(dropout_idxs.sum()), 1))

        input = einops.rearrange(input, " (b outer inner mod) k -> b outer inner mod k", b=input_shape[0], outer=input_shape[1], inner=input_shape[2], mod=input_shape[3])

        return input


class My_MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 dim_proj = 128,
                 activation=F.relu,
                 gbiased = None,
                 rpos = False
                 ):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(My_MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.gbiased = gbiased
        self.linear_q = nn.Linear(in_features, dim_proj, bias)
        self.linear_k = nn.Linear(in_features, dim_proj, bias)
        self.linear_v = nn.Linear(in_features, dim_proj, bias)
        self.linear_o = nn.Linear(dim_proj, in_features, False)

        self.scaled_dotproduct_attention =  ScaledDotProductAttention( rpos=rpos, d_head=int(dim_proj / head_num), head_num=head_num)

    def forward(self, q, prev, k, attn_mask=None, key_padding_mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(prev)

        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)


        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.head_num, 1, 1)
        y, att = self.scaled_dotproduct_attention(q, k, v, self.gbiased, prevalue=prev, mask=attn_mask)

        y = self._reshape_from_batches(y)
        y = self.linear_o(y)

        if self.activation is not None:
            y = self.activation(y)

        return y, att, v, self.linear_o

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        seq_len, batch_size, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return einops.rearrange(x, "seq b (h sub_dim)-> seq (b h) sub_dim", h=self.head_num, sub_dim=sub_dim)


    def _reshape_from_batches(self, x):
        seq_len, batch_size, in_feature = x.size()
        batch_size //= self.head_num
        return einops.rearrange(x, "seq (b h) sub_dim -> seq b (h sub_dim)", h=self.head_num, b=batch_size)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )
class My_MultiHeadAttention_HP(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 dim_proj = 128,
                 activation=F.relu,
                 gbiased = None,
                 rpos = False
                 ):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(My_MultiHeadAttention_HP, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.gbiased = gbiased

        self.linear_q = nn.Linear(in_features, dim_proj, bias)
        self.linear_k = nn.Linear(in_features, dim_proj, bias)
        self.linear_v = nn.Linear(in_features, dim_proj, bias)
        self.linear_o = nn.Linear(dim_proj, in_features, False)

        self.scaled_dotproduct_attention =  ScaledDotProductAttention_HP( rpos=rpos, d_head=int(in_features / head_num), head_num=head_num)

    def forward(self, q, k, prev, attn_mask=None, key_padding_mask=None):

        q_eeg, k_eeg, v_eeg = self.linear_q(q[0]), self.linear_k(k[0]), self.linear_v(prev[0])
        q_eog, k_eog, v_eog = self.linear_q(q[1]), self.linear_k(k[1]), self.linear_v(prev[1])

        if self.activation is not None:
            q_eeg = self.activation(q_eeg)
            q_eog = self.activation(q_eog)
            k_eeg = self.activation(k_eeg)
            k_eog = self.activation(k_eog)
            v_eeg = self.activation(v_eeg)
            v_eog = self.activation(v_eog)

        q_eeg = self._reshape_to_batches(q_eeg)
        q_eog = self._reshape_to_batches(q_eog)
        k_eeg = self._reshape_to_batches(k_eeg)
        k_eog = self._reshape_to_batches(k_eog)
        v_eeg = self._reshape_to_batches(v_eeg)
        v_eog = self._reshape_to_batches(v_eog)


        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.head_num, 1, 1)
        y, att = self.scaled_dotproduct_attention([q_eeg, q_eog], [k_eeg, k_eog], [v_eeg, v_eog], self.gbiased, prevalue=prev, mask=attn_mask)

        y_eeg = self._reshape_from_batches(y[0])
        y_eog = self._reshape_from_batches(y[1])
        y_eeg = self.linear_o(y_eeg)
        y_eog = self.linear_o(y_eog)

        if self.activation is not None:
            y_eeg = self.activation(y_eeg)
            y_eog = self.activation(y_eog)

        return [y_eeg, y_eog], att, [v_eeg, v_eog], self.linear_o

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        seq_len, batch_size, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return einops.rearrange(x, "seq b (h sub_dim)-> seq (b h) sub_dim", h=self.head_num, sub_dim=sub_dim)


    def _reshape_from_batches(self, x):
        seq_len, batch_size, in_feature = x.size()
        batch_size //= self.head_num
        return einops.rearrange(x, "seq (b h) sub_dim -> seq b (h sub_dim)", h=self.head_num, b=batch_size)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )
class My_MultiHeadAttention_Conv(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 dim_proj = 128,
                 activation=F.relu,
                 gbiased = None,
                 rpos = False
                 ):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(My_MultiHeadAttention_Conv, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.gbiased = gbiased
        self.linear_q = nn.Conv1d(in_features,in_features,3, padding=1)
        self.linear_k = nn.Conv1d(in_features,in_features,3, padding=1)
        self.linear_v = nn.Linear(in_features, dim_proj, bias)
        self.linear_o = nn.Linear(dim_proj, in_features, False)

        self.scaled_dotproduct_attention =  ScaledDotProductAttention( rpos=rpos, d_head=int(in_features / head_num), head_num=head_num)

    def forward(self, q, k, prev, attn_mask=None, key_padding_mask=None):

        q = einops.rearrange(q, "seq b f-> b f seq")
        k = einops.rearrange(k, "seq b f-> b f seq")
        q, k, = self.linear_q(q), self.linear_k(k)
        k = einops.rearrange(k, "b f seq -> seq b f")
        q = einops.rearrange(q, "b f seq -> seq b f")

        v = self.linear_v(prev)

        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)


        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.head_num, 1, 1)
        y, att = self.scaled_dotproduct_attention(q, k, v, self.gbiased, prevalue=prev, mask=attn_mask)

        y = self._reshape_from_batches(y)
        y = self.linear_o(y)

        if self.activation is not None:
            y = self.activation(y)

        return y, att, v, self.linear_o

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        seq_len, batch_size, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return einops.rearrange(x, "seq b (h sub_dim)-> seq (b h) sub_dim", h=self.head_num, sub_dim=sub_dim)


    def _reshape_from_batches(self, x):
        seq_len, batch_size, in_feature = x.size()
        batch_size //= self.head_num
        return einops.rearrange(x, "seq (b h) sub_dim -> seq b (h sub_dim)", h=self.head_num, b=batch_size)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )

class Multiply_Att(nn.Module):
    def forward(self, v, att):
        out = torch.einsum('b i j , j b d ->  i b d', att, v)
        return out

class My_MultiHeadAttention_Norm_RA(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 dim_proj = 128,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(My_MultiHeadAttention_Norm_RA, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, dim_proj, bias)
        self.linear_k = nn.Linear(in_features, dim_proj, bias)
        self.linear_v = nn.Linear(in_features, dim_proj, bias)
        self.wo = nn.Parameter(torch.Tensor(dim_proj, in_features))
        init.kaiming_uniform_(self.wo, a=math.sqrt(5))
        self.scaled_dotproduct_attention =  ScaledDotProductAttention()
        self.multiply_att = Multiply_Att()

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.head_num, 1, 1)
        yout, att = self.scaled_dotproduct_attention(q, k, v, attn_mask)

        v = self._reshape_from_batches(v)
        y = einops.rearrange(v, 'i b d -> b i d ')
        y = torch.einsum('bsi,im->bsm ', y, self.wo)
        y = einops.rearrange(y, 'i b d -> b i d ')
        y = self._reshape_to_batches(y)
        y = self.multiply_att(y, att)
        y = self._reshape_from_batches(y)

        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        seq_len, batch_size, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return einops.rearrange(x, "seq b (h sub_dim)-> seq (b h) sub_dim", h=self.head_num, sub_dim=sub_dim)


    def _reshape_from_batches(self, x):
        seq_len, batch_size, in_feature = x.size()
        batch_size //= self.head_num
        return einops.rearrange(x, "seq (b h) sub_dim -> seq b (h sub_dim)", h=self.head_num, b=batch_size)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )
class My_MultiHeadAttention_Sparse(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 dim_proj = 128,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(My_MultiHeadAttention_Sparse, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, dim_proj, bias)
        self.linear_qs = nn.Linear(in_features, dim_proj, bias)
        self.linear_k = nn.Linear(in_features, dim_proj, bias)
        self.linear_ks = nn.Linear(in_features, dim_proj, bias)
        self.linear_v = nn.Linear(in_features, dim_proj, bias)
        self.linear_o = nn.Linear(dim_proj, in_features, bias)

        self.scaled_dotproduct_attention =  ScaledDotProductAttention_Sparse()

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        qf, kf, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        qs, ks = self.linear_qs(q), self.linear_ks(k)


        if self.activation is not None:
            q = self.activation(q)
            qs = self.activation(qs)
            k = self.activation(k)
            ks = self.activation(ks)
            v = self.activation(v)

        q = self._reshape_to_batches(qf)
        qs = self._reshape_to_batches(qs)
        k = self._reshape_to_batches(kf)
        ks = self._reshape_to_batches(ks)
        v = self._reshape_to_batches(v)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.head_num, 1, 1)
        y, att = self.scaled_dotproduct_attention(q, k, v, qs, ks, attn_mask)
        y = self._reshape_from_batches(y)
        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        seq_len, batch_size, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return einops.rearrange(x, "seq b (h sub_dim)-> seq (b h) sub_dim", h=self.head_num, sub_dim=sub_dim)


    def _reshape_from_batches(self, x):
        seq_len, batch_size, in_feature = x.size()
        batch_size //= self.head_num
        return einops.rearrange(x, "seq (b h) sub_dim -> seq b (h sub_dim)", h=self.head_num, b=batch_size)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )
class My_TF(nn.Module):
    def __init__(self, d_model, nhead, modalities=1, dim_feedforward=1024, dim_proj= 1024, dropout=0.1, activation="relu"):
        # super(nn.TransformerEncoderLayer, self).__init__()
        super().__init__()
        # self.self_attn = My_MultiHeadAttention(d_model,  nhead, dim_proj=dim_proj, activation=None)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.mod_0_linear1 = nn.Linear(d_model, dim_feedforward)
        self.mod_0_dropout = nn.Dropout(dropout)
        self.mod_0_linear2 = nn.Linear(dim_feedforward, d_model)
        self.mod_0_norm2 = nn.LayerNorm(d_model)
        self.mod_0_dropout2 = nn.Dropout(dropout)

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

        # self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        # src = einops.rearrange(src, "b outer inner mod k -> (inner mod) (b outer) k")

        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # src = einops.rearrange(src, "(inner mod) b_outer k -> mod inner b_outer k", inner = self.inner, mod = self.mod)

        src2 = self.mod_0_linear2(self.mod_0_dropout(self.activation(self.mod_0_linear1(src))))
        src = src + self.mod_0_dropout2(src2)
        src = self.mod_0_norm2(src)

        # src2 = self.mod_1_linear2(self.mod_1_dropout(self.activation(self.mod_1_linear1(src[1]))))
        # src[1] = src[0] + self.mod_1_dropout2(src2)
        # src[1] = self.mod_1_norm2(src[1])
        #
        # src = einops.rearrange(src, "mod inner (b outer) k -> b outer inner mod k",b=self.batch, outer=self.outer, mod=self.mod)

        return src

class My_TF_RA(nn.Module):
    def __init__(self, d_model, nhead, gbiased=False, extra_attention = False, rpos=False, modalities=1, dim_feedforward=1024, dim_proj= 128, dropout=0.1, activation="relu", ln_first=False):
        # super(nn.TransformerEncoderLayer, self).__init__()
        super().__init__()

        self.ln_first = ln_first

        self.extra_attention = extra_attention
        if self.extra_attention:
            self.extra_self_attn = My_MultiHeadAttention(d_model, nhead, dim_proj=128, activation=None, gbiased=self.extra_attention)
            self.extra_norm = nn.LayerNorm(d_model)
            self.extra_dropout = nn.Dropout(dropout)

        self.self_attn_my = My_MultiHeadAttention(d_model,  nhead, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)

        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.mod_0_linear1 = nn.Linear(d_model, dim_feedforward)
        self.mod_0_dropout = nn.Dropout(dropout)
        self.mod_0_linear2 = nn.Linear(dim_feedforward, d_model)
        self.mod_0_norm2 = nn.LayerNorm(d_model)
        self.mod_0_dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError('Activation {} does not exist! Available options are "relu", "gelu".')
        self.norm_calc = BertNormOutput(num_attention_heads=nhead, hidden_size=d_model)
        self.nhead = nhead


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, extract_norm = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x_shape = src.shape

        if self.extra_attention:
            if self.ln_first: src = self.extra_norm(src)
            src_extr_att, att_0, value_0, linear_o_0 = self.extra_self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            src_extr_att = src + self.extra_dropout(src_extr_att)
            if not self.ln_first: src_extr_att = self.extra_norm(src_extr_att)
        else:
            src_extr_att = src

        if self.ln_first: src_extr_att = self.norm1(src_extr_att)
        src_att, att, value, linear_o = self.self_attn_my(src_extr_att, src_extr_att, src_extr_att, attn_mask=src_mask,  key_padding_mask=src_key_padding_mask)
        src_att_drop_norm = src_extr_att + self.dropout1(src_att)
        if not self.ln_first: src_att_drop_norm = self.norm1(src_att_drop_norm)

        if self.ln_first: src_att_drop_norm = self.mod_0_norm2(src_att_drop_norm)
        src_att_drop_norm_fc = self.mod_0_linear2(self.mod_0_dropout(self.activation(self.mod_0_linear1(src_att_drop_norm))))
        src_att_drop_norm_fc_drop_nrom = src_att_drop_norm + self.mod_0_dropout2(src_att_drop_norm_fc)
        if not self.ln_first: src_att_drop_norm_fc_drop_nrom = self.mod_0_norm2(src_att_drop_norm_fc_drop_nrom)


        if extract_norm:
            batch_size = int(src.shape[1])
            vector_norms = self.norm_calc(hidden_states=einops.rearrange(src, " s b f -> b s f"),
                               attention_probs=einops.rearrange(att, "(b h) i j -> b h i j", b=batch_size, h=self.nhead),
                               value_layer=einops.rearrange(value, "seq (b h) fh -> b h seq fh", b=batch_size, h=self.nhead),
                               dense=linear_o, LayerNorm=self.norm1,
                               pre_ln_states=einops.rearrange(src_att_drop_norm, " s b f -> b s f"))

        return src_att_drop_norm_fc_drop_nrom
class My_TF_Decoder_RA(nn.Module):
    def __init__(self, d_model, nhead, gbiased=False, extra_attention = False, rpos=False, modalities=1, dim_feedforward=1024, dim_proj= 128, dropout=0.1, activation="relu"):
        # super(nn.TransformerEncoderLayer, self).__init__()
        super().__init__()

        self.extra_attention = extra_attention
        if self.extra_attention:
            self.extra_self_attn = My_MultiHeadAttention(d_model, nhead, dim_proj=128, activation=None, gbiased=self.extra_attention)
            self.extra_norm = nn.LayerNorm(d_model)
            self.extra_dropout = nn.Dropout(dropout)

        self.self_attn_my = My_MultiHeadAttention(d_model,  nhead, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)

        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.mod_0_linear1 = nn.Linear(d_model, dim_feedforward)
        self.mod_0_dropout = nn.Dropout(dropout)
        self.mod_0_linear2 = nn.Linear(dim_feedforward, d_model)
        self.mod_0_norm3 = nn.LayerNorm(d_model)
        self.mod_0_dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.norm_calc = BertNormOutput(num_attention_heads=nhead, hidden_size=d_model)
        self.nhead = nhead


    def forward(self, output: Tensor, input: Tensor,  output_mask: Optional[Tensor] = None, input_mask: Optional[Tensor] = None, extract_norm = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x_shape = output.shape

        if self.extra_attention:
            output_extr_att, att_0, value_0, linear_o_0 = self.extra_self_attn(output, output, output, attn_mask=output_mask)
            output_extr_att = self.extra_norm(output + self.extra_dropout(output_extr_att))
        else:
            output_extr_att = output

        output_att, att, value, linear_o = self.self_attn_my(output_extr_att, output_extr_att, output_extr_att, attn_mask=output_mask)
        output_att = self.norm1(output_extr_att + self.dropout1(output_att))

        input_output_att, att, value, linear_o = self.self_attn_my(output_att, input, input, attn_mask=output_mask)
        input_output_att = self.norm2(input_output_att + self.dropout2(output_att))

        input_output_att_drop_norm_fc = self.mod_0_linear2(self.mod_0_dropout(self.activation(self.mod_0_linear1(input_output_att))))
        src_att_drop_norm_fc_drop_norm = self.mod_0_norm3(input_output_att + self.mod_0_dropout3(input_output_att_drop_norm_fc))

        # if extract_norm:
        #     batch_size = int(src.shape[1])
        #     vector_norms = self.norm_calc(hidden_states=einops.rearrange(src, " s b f -> b s f"),
        #                        attention_probs=einops.rearrange(att, "(b h) i j -> b h i j", b=batch_size, h=self.nhead),
        #                        value_layer=einops.rearrange(value, "seq (b h) fh -> b h seq fh", b=batch_size, h=self.nhead),
        #                        dense=linear_o, LayerNorm=self.norm1,
        #                        pre_ln_states=einops.rearrange(src_att_drop_norm, " s b f -> b s f"))

        return src_att_drop_norm_fc_drop_norm
class My_TF_Conv_RA(nn.Module):
    def __init__(self, d_model, nhead, gbiased=False, extra_attention = False, rpos=False, modalities=1, dim_feedforward=1024, dim_proj= 128, dropout=0.1, activation="relu"):
        # super(nn.TransformerEncoderLayer, self).__init__()
        super().__init__()

        self.extra_attention = extra_attention
        if self.extra_attention:
            self.extra_self_attn = My_MultiHeadAttention_Conv(d_model, nhead, dim_proj=128, activation=None, gbiased=self.extra_attention)
            self.extra_norm = nn.LayerNorm(d_model)
            self.extra_dropout = nn.Dropout(dropout)

        self.self_attn_my = My_MultiHeadAttention_Conv(d_model,  nhead, dim_proj=dim_proj, rpos=rpos, activation=None, gbiased = gbiased)

        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.mod_0_linear1 = nn.Linear(d_model, dim_feedforward)
        self.mod_0_dropout = nn.Dropout(dropout)
        self.mod_0_linear2 = nn.Linear(dim_feedforward, d_model)
        self.mod_0_norm2 = nn.LayerNorm(d_model)
        self.mod_0_dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.norm_calc = BertNormOutput(num_attention_heads=nhead, hidden_size=d_model)
        self.nhead = nhead


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, extract_norm = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x_shape = src.shape

        if self.extra_attention:
            src_extr_att, att_0, value_0, linear_o_0 = self.extra_self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            src_extr_att = src + self.extra_dropout(src_extr_att)
            src_extr_att = self.extra_norm(src_extr_att)
        else:
            src_extr_att = src

        src_att, att, value, linear_o = self.self_attn_my(src_extr_att, src_extr_att, src_extr_att, attn_mask=src_mask,  key_padding_mask=src_key_padding_mask)

        src_att_drop = src_extr_att + self.dropout1(src_att)
        src_att_drop_norm = self.norm1(src_att_drop)

        src_att_drop_norm_fc = self.mod_0_linear2(self.mod_0_dropout(self.activation(self.mod_0_linear1(src_att_drop_norm))))
        src_att_drop_norm_fc_drop = src_att_drop_norm + self.mod_0_dropout2(src_att_drop_norm_fc)
        src_att_drop_norm_fc_drop_nrom = self.mod_0_norm2(src_att_drop_norm_fc_drop)

        if extract_norm:
            batch_size = int(src.shape[1])
            vector_norms = self.norm_calc(hidden_states=einops.rearrange(src, " s b f -> b s f"),
                               attention_probs=einops.rearrange(att, "(b h) i j -> b h i j", b=batch_size, h=self.nhead),
                               value_layer=einops.rearrange(value, "seq (b h) fh -> b h seq fh", b=batch_size, h=self.nhead),
                               dense=linear_o, LayerNorm=self.norm1,
                               pre_ln_states=einops.rearrange(src_att_drop_norm, " s b f -> b s f"))

        return src_att_drop_norm_fc_drop_nrom
class My_TF_normreg_RA(nn.Module):
    def __init__(self, d_model, nhead, gbiased=False, extra_attention = False, rpos=False, modalities=1, dim_feedforward=1024, dim_proj= 128, dropout=0.1, activation="relu"):
        # super(nn.TransformerEncoderLayer, self).__init__()
        super().__init__()

        self.extra_attention = extra_attention
        if self.extra_attention:
            self.extra_self_attn = My_MultiHeadAttention(d_model, nhead, dim_proj=128, activation=None, gbiased=self.extra_attention)
            self.extra_norm = nn.LayerNorm(d_model)
            self.extra_dropout = nn.Dropout(dropout)

        self.self_attn_my = My_MultiHeadAttention(d_model,  nhead, dim_proj=128, rpos=rpos, activation=None, gbiased = gbiased)

        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.mod_0_linear1 = nn.Linear(d_model, dim_feedforward)
        self.mod_0_dropout = nn.Dropout(dropout)
        self.mod_0_linear2 = nn.Linear(dim_feedforward, d_model)
        self.mod_0_norm2 = nn.LayerNorm(d_model)
        self.mod_0_dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.norm_calc = BertNormOutput(num_attention_heads=nhead, hidden_size=d_model)
        self.nhead = nhead


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, extract_norm = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x_shape = src.shape

        if self.extra_attention:
            src_extr_att, att_0, value_0, linear_o_0 = self.extra_self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            src_extr_att = src + self.extra_dropout(src_extr_att)
            src_extr_att = self.extra_norm(src_extr_att)
        else:
            src_extr_att = src

        src_att, att, value, linear_o = self.self_attn_my(src_extr_att, src_extr_att, src_extr_att, attn_mask=src_mask,  key_padding_mask=src_key_padding_mask)
        norm_att = torch.norm(src_att)
        norm_prev = torch.norm(src_extr_att)

        src_att_drop = src_extr_att + (norm_prev/norm_att)*self.dropout1(src_att)

        src_att_drop_norm = self.norm1(src_att_drop)

        src_att_drop_norm_fc = self.mod_0_linear2(self.mod_0_dropout(self.activation(self.mod_0_linear1(src_att_drop_norm))))
        src_att_drop_norm_fc_drop = src_att_drop_norm + self.mod_0_dropout2(src_att_drop_norm_fc)
        src_att_drop_norm_fc_drop_nrom = self.mod_0_norm2(src_att_drop_norm_fc_drop)

        if extract_norm:
            batch_size = int(src.shape[1])
            vector_norms = self.norm_calc(hidden_states=einops.rearrange(src, " s b f -> b s f"),
                               attention_probs=einops.rearrange(att, "(b h) i j -> b h i j", b=batch_size, h=self.nhead),
                               value_layer=einops.rearrange(value, "seq (b h) fh -> b h seq fh", b=batch_size, h=self.nhead),
                               dense=linear_o, LayerNorm=self.norm1,
                               pre_ln_states=einops.rearrange(src_att_drop_norm, " s b f -> b s f"))

        return src_att_drop_norm_fc_drop_nrom
class My_TF_LSTM(nn.Module):
    def __init__(self, d_model, nhead, gbiased=False, extra_attention = False, rpos=False, modalities=1, dim_feedforward=1024, dim_proj= 128, dropout=0.1, activation="relu"):
        # super(nn.TransformerEncoderLayer, self).__init__()
        super().__init__()

        self.extra_attention = extra_attention
        if self.extra_attention:
            self.extra_self_attn = My_MultiHeadAttention(d_model, nhead, dim_proj=128, activation=None, gbiased=self.extra_attention)
            self.extra_norm = nn.LayerNorm(d_model)
            self.extra_dropout = nn.Dropout(dropout)

        self.self_attn_my = My_MultiHeadAttention(d_model,  nhead, dim_proj=128, rpos=rpos, activation=None, gbiased = gbiased)

        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.mod_0_norm2 = nn.LayerNorm(d_model)
        self.mod_0_dropout2 = nn.Dropout(dropout)

        self.lstm = nn.LSTM(d_model, hidden_size = d_model, num_layers= 1, bidirectional=False)

        self.norm_calc = BertNormOutput(num_attention_heads=nhead, hidden_size=d_model)
        self.nhead = nhead


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, extract_norm = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x_shape = src.shape

        if self.extra_attention:
            src_extr_att, att_0, value_0, linear_o_0 = self.extra_self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            src_extr_att = src + self.extra_dropout(src_extr_att)
            src_extr_att = self.extra_norm(src_extr_att)
        else:
            src_extr_att = src

        src_att, att, value, linear_o = self.self_attn_my(src_extr_att, src_extr_att, src_extr_att, attn_mask=src_mask,  key_padding_mask=src_key_padding_mask)

        src_att_drop = src_extr_att + self.dropout1(src_att)
        src_att_drop_norm = self.norm1(src_att_drop)

        src_att_drop_norm_fc = self.lstm(src_att_drop_norm)[0]
        src_att_drop_norm_fc_drop = src_att_drop_norm + self.mod_0_dropout2(src_att_drop_norm_fc)
        src_att_drop_norm_fc_drop_nrom = self.mod_0_norm2(src_att_drop_norm_fc_drop)

        if extract_norm:
            batch_size = int(src.shape[1])
            vector_norms = self.norm_calc(hidden_states=einops.rearrange(src, " s b f -> b s f"),
                               attention_probs=einops.rearrange(att, "(b h) i j -> b h i j", b=batch_size, h=self.nhead),
                               value_layer=einops.rearrange(value, "seq (b h) fh -> b h seq fh", b=batch_size, h=self.nhead),
                               dense=linear_o, LayerNorm=self.norm1,
                               pre_ln_states=einops.rearrange(src_att_drop_norm, " s b f -> b s f"))

        return src_att_drop_norm_fc_drop_nrom
class My_TF_HPFC_RA(nn.Module):
    def __init__(self, d_model, nhead, gbiased=False, modalities=1, rpos=False, extra_attention = False, dim_feedforward=1024, dim_proj= 1024, dropout=0.1, activation="relu"):
        # super(nn.TransformerEncoderLayer, self).__init__()
        super().__init__()

        self.self_attn_my = My_MultiHeadAttention_HP(d_model,  nhead, dim_proj=128, activation=None, gbiased = gbiased)

        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.modalities = modalities

        # self.mod_0_linear1 = nn.Linear(d_model, dim_feedforward)
        # self.mod_0_dropout = nn.Dropout(dropout)
        # self.mod_0_linear2 = nn.Linear(dim_feedforward, d_model)
        # self.mod_0_dropout2 = nn.Dropout(dropout)
        for i in range(modalities):
            setattr(self,"mod{}_fc".format(i),nn.Sequential( nn.Linear(d_model, dim_feedforward),
                                           self.activation,
                                           nn.Dropout(dropout),
                                           nn.Linear(dim_feedforward, d_model),
                                           nn.Dropout(dropout)
                                           ))
            setattr(self,"mod{}_norm1".format(i), nn.LayerNorm(d_model))
            setattr(self,"mod{}_norm2".format(i), nn.LayerNorm(d_model))
            setattr(self,"mod{}_dropout".format(i), nn.Dropout(dropout))

        self.norm_calc = BertNormOutput(num_attention_heads=nhead, hidden_size=d_model)
        self.nhead = nhead


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, extract_norm = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x_shape = src[0].shape

        src_att, att, value, linear_o = self.self_attn_my(src, src, src, attn_mask=src_mask,  key_padding_mask=src_key_padding_mask)

        # src_att_drop = src + self.dropout1(src_att)
        # src_att_drop_norm = self.norm1(src_att_drop)
        #
        # seq_mod_split = int(x_shape[0]/self.modalities)
        src_att_drop = []
        for i in range(self.modalities):
            fc = getattr(self, "mod{}_fc".format(i))
            norm1 = getattr(self, "mod{}_norm1".format(i))
            norm2 = getattr(self, "mod{}_norm2".format(i))
            dropout1 = getattr(self, "mod{}_dropout".format(i))
            src_att[i] = src[i] + dropout1(src_att[i])
            src_att[i] = norm1(src_att[i])
            src_att[i] = fc(src_att[i]) + src_att[i]
            src_att[i] = norm2(src_att[i])

        # if self.modalities>1:
        #     src_att_output = torch.cat([locals()[ "src_att_norm_mod0"], locals()["src_att_norm_mod1"]],dim=0)
        # else:
        #     src_att_output = locals()["src_att_norm_mod0"]
        # if extract_norm:
        #     batch_size = int(src.shape[1])
        #     norms = [getattr(self, "mod{}_norm".format(i)) for i in range(self.modalities)]
        #
        #     vector_norms = self.norm_calc(hidden_states=einops.rearrange(src, " s b f -> b s f"),
        #                        attention_probs=einops.rearrange(att, "(b h) i j -> b h i j", b=batch_size, h=self.nhead),
        #                        value_layer=einops.rearrange(value, "seq (b h) fh -> b h seq fh", b=batch_size, h=self.nhead),
        #                        dense=linear_o, LayerNorm=self.norm1,
        #                        pre_ln_states=einops.rearrange(src_att_output, " s b f -> b s f"))

        return src_att

class My_TF_Sparse_RA(nn.Module):
    def __init__(self, d_model, nhead, modalities=1, dim_feedforward=1024, dim_proj= 1024, dropout=0.1, activation="relu"):
        # super(nn.TransformerEncoderLayer, self).__init__()
        super().__init__()
        self.self_attn_my = My_MultiHeadAttention_Sparse(d_model,  nhead, dim_proj=dim_proj, activation=None)
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.mod_0_linear1 = nn.Linear(d_model, dim_feedforward)
        self.mod_0_dropout = nn.Dropout(dropout)
        self.mod_0_linear2 = nn.Linear(dim_feedforward, d_model)
        self.mod_0_norm2 = nn.LayerNorm(d_model)
        self.mod_0_dropout2 = nn.Dropout(dropout)

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

        # self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        # src = einops.rearrange(src, "b outer inner mod k -> (inner mod) (b outer) k")
        # src2_0 = self.self_attn(src, src, src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]

        src2 = self.self_attn_my(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        # print(src2_0[0,0,0:10])
        # print(src2[0,0,0:10])

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # src = einops.rearrange(src, "(inner mod) b_outer k -> mod inner b_outer k", inner = self.inner, mod = self.mod)

        src2 = self.mod_0_linear2(self.mod_0_dropout(self.activation(self.mod_0_linear1(src))))
        src = src + self.mod_0_dropout2(src2)
        src = self.mod_0_norm2(src)

        # src2 = self.mod_1_linear2(self.mod_1_dropout(self.activation(self.mod_1_linear1(src[1]))))
        # src[1] = src[0] + self.mod_1_dropout2(src2)
        # src[1] = self.mod_1_norm2(src[1])
        #
        # src = einops.rearrange(src, "mod inner (b outer) k -> b outer inner mod k",b=self.batch, outer=self.outer, mod=self.mod)

        return src

class My_TF_diff_fc(nn.Module):
    def __init__(self, d_model, nhead, modalities=2, dim_feedforward=1024, dropout=0.1, activation="relu"):
        # super(nn.TransformerEncoderLayer, self).__init__()
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.mod_0_linear1 = nn.Linear(d_model, dim_feedforward)
        self.mod_0_dropout = nn.Dropout(dropout)
        self.mod_0_linear2 = nn.Linear(dim_feedforward, d_model)
        self.mod_0_norm2 = nn.LayerNorm(d_model)
        self.mod_0_dropout2 = nn.Dropout(dropout)

        self.mod_1_linear1 = nn.Linear(d_model, dim_feedforward)
        self.mod_1_dropout = nn.Dropout(dropout)
        self.mod_1_linear2 = nn.Linear(dim_feedforward, d_model)
        self.mod_1_norm2 = nn.LayerNorm(d_model)
        self.mod_1_dropout2 = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

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

        mod, inner = src.shape[0], src.shape[1]
        src = einops.rearrange(src, "mod inner b_outer k -> (inner mod) b_outer k ")

        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = einops.rearrange(src, "(inner mod) b_outer k  -> mod inner b_outer k", mod=mod, inner=inner)

        src2 = self.mod_0_linear2(self.mod_0_dropout(self.activation(self.mod_0_linear1(src[0]))))
        src[0] = src[0] + self.mod_0_dropout2(src2)
        src[0] = self.mod_0_norm2(src[0])

        src2 = self.mod_1_linear2(self.mod_1_dropout(self.activation(self.mod_1_linear1(src[1]))))
        src[1] = src[0] + self.mod_1_dropout2(src2)
        src[1] = self.mod_1_norm2(src[1])

        return src

class TF_outer_mod_att_inner_diff_fc(nn.Module):
    def __init__(self, d_model, nhead, modalities=1, dim_feedforward=1024, dropout=0.1, activation="relu"):
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

        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        src = einops.rearrange(src, "b outer inner mod k -> (outer mod) b (inner k)")

        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = einops.rearrange(src, "(outer mod) b inner_k -> mod outer b inner_k", outer = self.outer, mod = self.mod)

        for i in range(len(src)):
            lin1 = getattr(self,"mod_{}_linear1".format(i))
            drop = getattr(self,"mod_{}_dropout".format(i))
            lin2 = getattr(self,"mod_{}_linear2".format(i))
            norm2 = getattr(self,"mod_{}_norm2".format(i))
            drop2 = getattr(self,"mod_{}_dropout2".format(i))
            src2 = lin2(drop(self.activation(lin1(src[i]))))
            src[i] = src[i] + drop2(src2)
            src[i] = norm2(src[i])

        src = einops.rearrange(src, "mod outer b (inner k) -> b outer inner mod k",b=self.batch, inner=self.inner,
                             mod=self.mod)
        return src

class inner_mod_att(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)
            self.pos_mod = PositionalEncoder(d_model=dmodel)

        enc = nn.TransformerEncoderLayer(dmodel, nhead=heads)
        self.inner_tf = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]


        x = einops.rearrange(x, "b outer inner mod k -> (inner mod) (b outer) k")
        if self.pos:
            x = einops.rearrange(x, "(inner mod) (b outer) k -> inner (b outer mod) k", mod=self.mod, outer = self.outer, b=self.batch, inner = self.inner)
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "inner (b outer mod) k -> mod (b outer inner) k", mod=self.mod, outer = self.outer, b=self.batch, inner = self.inner)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "mod (b outer inner) k -> (inner mod) (b outer) k", mod=self.mod, outer = self.outer, b=self.batch, inner = self.inner)
        x = self.inner_tf(x)
        x = einops.rearrange(x, "(inner mod) (b outer) k -> b outer inner mod k", mod=self.mod, outer = self.outer, b=self.batch, inner = self.inner)
        return x
class inner_outer_cross_att(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)
            self.pos_mod = PositionalEncoder(d_model=dmodel)

        self.num_layers = num_layers
        for i in range(num_layers):
            setattr(self, "outter_inner_cross_att_{}".format(i), MyViLBERT(dmodel, nheads=heads))

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
            x = einops.rearrange(x, "outer (b inner mod) k -> b outer inner mod k", mod=self.mod, inner=self.inner, b = self.b)

        x = einops.rearrange(x, "b outer inner mod ch k -> (outer inner) (mod ch) b k")
        x0 = x[:,0,:,:]
        x1 = x[:,1,:,:]
        for i in range(self.num_layers):
            layer = getattr(self, "outter_inner_cross_att_{}".format(i))
            x0, x1 = layer(x0, x1)
        x0 = einops.rearrange(x0, "(outer inner mod) b k -> (outer inner) mod b k", outer=self.outer, inner=self.inner, mod=1)
        x1 = einops.rearrange(x1, "(outer inner mod) b k -> (outer inner) mod b k", outer=self.outer, inner=self.inner, mod=1)
        x = torch.cat([x0,x1], dim=1)
        x = einops.rearrange(x, "(outer inner) (mod ch) b k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,
                             b=self.batch)
        return x
class inner_cross_att(nn.Module):
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
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        if self.pos:
            x = einops.rearrange(x, "b outer inner mod k -> inner (b outer mod) k")
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "inner (b outer mod) k -> mod (b outer inner) k", mod=self.mod)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "mod (b outer inner) k -> outer (b inner mod) k", outer=self.outer)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "outer (b inner mod) k -> b outer inner mod k", mod=self.mod, inner=self.inner, b = self.b)

        x = einops.rearrange(x, "b outer inner mod k -> inner mod (outer b) k")
        x0 = x[:,0,:,:]
        x1 = x[:,1,:,:]
        for i in range(self.num_layers):
            layer = getattr(self, "inner_cross_att_{}".format(i))
            x0, x1 = layer(x0, x1)
        x0 = einops.rearrange(x0, "(inner mod) b k -> inner mod b k", inner=self.inner, mod=1)
        x1 = einops.rearrange(x1, "(inner mod) b k -> inner mod b k", inner=self.inner, mod=1)
        x = torch.cat([x0,x1], dim=1)
        x = einops.rearrange(x, "inner mod (outer b) k -> b outer inner mod k", outer=self.outer, mod=self.mod, b=self.batch)
        return x
class outer_cross_att(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
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
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]


        if self.pos:
            x = einops.rearrange(x, "b outer inner mod k -> outer (b outer mod) k")
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "inner (b outer mod) k -> mod (b outer inner) k", mod=self.mod)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "mod (b outer inner) k -> outer (b inner mod) k", outer=self.outer)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "outer (b inner mod) k -> b outer inner mod k", mod=self.mod, inner=self.inner,
                                 b=self.b)

        x = einops.rearrange(x, "b outer inner mod k -> outer mod (inner b) k")
        x0 = x[:, 0, :, :]
        x1 = x[:, 1, :, :]
        for i in range(self.num_layers):
            layer = getattr(self, "outter_cross_att_{}".format(i))
            x0, x1 = layer(x0, x1)
        x0 = einops.rearrange(x0, "(outer mod) b k -> outer mod b k", outer=self.outer, mod=1)
        x1 = einops.rearrange(x1, "(outer mod) b k -> outer mod b k", outer=self.outer, mod=1)
        x = torch.cat([x0, x1], dim=1)
        x = einops.rearrange(x, "outer mod (inner b) k -> b outer inner mod k", inner=self.inner, mod=self.mod,
                             b=self.batch)
        return x
class inner_mod_outer_att(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)
            self.pos_mod = PositionalEncoder(d_model=dmodel)

        enc = nn.TransformerEncoderLayer(dmodel, nhead=heads)
        self.all_tf = nn.TransformerEncoder(enc, num_layers)

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
            x = einops.rearrange(x, "outer (b inner mod) k -> b outer inner mod k", mod=self.mod, inner=self.inner, b = self.b)

        x = einops.rearrange(x, "b outer inner mod k -> (outer inner mod) b k")
        x = self.all_tf(x)
        x = einops.rearrange(x, "(outer inner mod) b k -> b outer inner mod k", outer=self.outer, mod=self.mod,
                             b=self.batch)
        return x
class outer_mod_att(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)
            self.pos_mod = PositionalEncoder(d_model=dmodel)

        enc = nn.TransformerEncoderLayer(dmodel, nhead=heads)
        self.all_tf = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]


        if self.pos:
            x = einops.rearrange(x, "b outer inner mod k -> inner (outer mod b) k")
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "inner (b outer mod) k -> mod (b outer inner) k", mod=self.mod, outer = self.outer, b=self.batch, inner = self.inner)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "mod (b outer inner) k -> outer (b inner mod) k", mod=self.mod, outer = self.outer, b=self.batch, inner = self.inner)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "outer (inner mod b) k -> b outer inner mod k", mod=self.mod, outer = self.outer, b=self.batch, inner = self.inner)

        x = einops.rearrange(x, "b outer inner mod k -> (outer mod) (b inner) k")
        x = self.all_tf(x)
        x = einops.rearrange(x, "(outer mod) (b inner) k -> b outer inner mod k", mod=self.mod, outer = self.outer, b=self.batch, inner = self.inner)
        return x
class inner_att(nn.Module):
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)
        def forward(self, x):
            x_shape = x.shape
            self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

            x = einops.rearrange(x, "b outer inner mod k -> inner (b outer mod) k")
            if self.pos:
                x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = self.inner_tf(x)
            x = einops.rearrange(x, "inner (b outer mod) k -> b outer inner mod k", outer=self.outer, mod=self.mod,
                                 b=self.batch)
            return x


# class inner_att_RA(nn.Module):
#     def __init__(self, dmodel, pos, inner, outer, modalities, gbiased=False, extra_attention=False, rpos=False, num_layers=1, dim_proj=128, heads=8, dim_feedforward=1024, ln_first=False):
#         super().__init__()
#         self.pos = pos
#         if pos:
#             self.pos_inner = PositionalEncoder(d_model=dmodel)
#
#         enc = My_TF_RA(dmodel, extra_attention=extra_attention, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased, ln_first=ln_first)
#         self.inner_tf = My_TransformerEncoder(enc, num_layers)
#
#     def forward(self, x, extract_norm=False):
#         x_shape = x.shape
#         self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]
#
#         x = einops.rearrange(x, "b outer inner mod ch k -> inner (b outer mod ch) k")
#         x = self.inner_tf(x, extract_norm=extract_norm)
#         x = einops.rearrange(x, "inner (b outer mod ch) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
#         return x
class inner_ch_att_HPFC_RA(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, gbiased=False, extra_attention=False, rpos=False, num_layers=1, dim_proj=128, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = My_TF_HPFC_RA(dmodel, extra_attention=extra_attention, modalities=modalities, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)
        self.inner_tf = My_TransformerEncoder(enc, num_layers)

    def forward(self, x, extract_norm=False):
        x_shape = x[0].shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x[0] = einops.rearrange(x[0], "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x[1] = einops.rearrange(x[1], "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x = self.inner_tf(x, extract_norm=extract_norm)
        x[0] = einops.rearrange(x[0], "(inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        x[1] = einops.rearrange(x[1], "(inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x
class inner_att_conv_RA(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, gbiased=False, extra_attention=False, rpos=False, num_layers=1, dim_proj=128, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = My_TF_Conv_RA(dmodel, extra_attention=extra_attention, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased)
        self.inner_tf = My_TransformerEncoder(enc, num_layers)

    def forward(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> inner (b outer mod ch) k")
        x = self.inner_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x, "inner (b outer mod ch) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

class inner_rn_RA(nn.Module):
    def __init__(self, dmodel, inner, outer, modalities, num_layers=1, bidirectional=False):
        super().__init__()

        self.inner_rn = nn.GRU(dmodel, dmodel, num_layers= num_layers, bidirectional=False)

    def forward(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> inner (b outer mod ch) k")
        _, x = self.inner_rn(x)
        x = x[-1:] #Last layer
        x = einops.rearrange(x, "inner (b outer mod ch) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

class inner_att_RA(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, dropout=0.1,  gbiased=False, extra_attention=False, rpos=False, num_layers=1, dim_proj=128, heads=8, dim_feedforward=1024, ln_first=False):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = My_TF_RA(dmodel, extra_attention=extra_attention, dropout=dropout, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased, ln_first=ln_first)
        self.inner_tf = My_TransformerEncoder(enc, num_layers)

    def forward(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> inner (b outer mod ch) k")
        x = self.inner_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x, " inner (b outer mod ch) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x
class inner_mod_ch_att_RA(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, dropout=0.1, gbiased=False, extra_attention=False, rpos=False, num_layers=1, dim_proj=128, heads=8, dim_feedforward=1024, ln_first=False):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = My_TF_RA(dmodel, dropout=dropout, extra_attention=extra_attention, nhead=heads, rpos=rpos, dim_proj=dim_proj, dim_feedforward=dim_feedforward, gbiased=gbiased, ln_first=ln_first)
        self.inner_tf = My_TransformerEncoder(enc, num_layers)

    def forward(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x = self.inner_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x, " (inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x
class inner_locglob_att_RA(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, gbiased=False, extra_attention=False, rpos=False, num_layers=1, heads=8, dim_feedforward=1024, ln_first=False):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = My_TF_RA(dmodel, extra_attention=extra_attention, nhead=heads, rpos=rpos, dim_feedforward=dim_feedforward, gbiased=gbiased, ln_first=ln_first)
        self.inner_tf_loc = My_TransformerEncoder(enc, num_layers)

        enc = My_TF_RA(dmodel, extra_attention=extra_attention, nhead=heads, rpos=rpos, dim_feedforward=dim_feedforward, gbiased=gbiased, ln_first=ln_first)
        self.inner_tf_glob = My_TransformerEncoder(enc, num_layers)

    def forward(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x, "b outer inner mod ch k -> inner (b outer mod ch) k")
        x = self.inner_tf_loc(x, extract_norm=extract_norm)
        x = einops.rearrange(x, " inner (b outer mod ch) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)

        x = einops.rearrange(x, "b outer inner mod ch k -> (inner mod ch) (b outer) k")
        x = self.inner_tf_glob(x, extract_norm=extract_norm)
        x = einops.rearrange(x, "(inner mod ch) (b outer) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, ch=self.ch,  b=self.batch)
        return x

class inner_att_normreg_RA(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, extra_attention=False, rpos=False, num_layers=1, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = My_TF_normreg_RA(dmodel, extra_attention=extra_attention, nhead=heads, rpos=rpos, dim_feedforward=dim_feedforward)
        self.inner_tf = My_TransformerEncoder(enc, num_layers)

    def forward(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod k -> inner (b mod outer) k")
        if self.pos:
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x, "inner (b mod outer) k -> b outer inner mod k", outer=self.outer, mod=self.mod,
                             b=self.batch)
        return x
class inner_intlstm(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, extra_attention=False, rpos=False, num_layers=1, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = My_TF_LSTM(dmodel, extra_attention=extra_attention, nhead=heads, rpos=rpos, dim_feedforward=dim_feedforward)
        self.inner_tf = My_TransformerEncoder(enc, num_layers)

    def forward(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod k -> inner (b mod outer) k")
        if self.pos:
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x, "inner (b mod outer) k -> b outer inner mod k", outer=self.outer, mod=self.mod,
                             b=self.batch)
        return x
class inner_att_fc_RA(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, extra_attention=False, rpos=False, num_layers=1, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = My_TF_FC_RA(dmodel, extra_attention=extra_attention, nhead=heads, rpos=rpos, modalities=modalities, dim_feedforward=dim_feedforward)
        self.inner_tf = My_TransformerEncoder(enc, num_layers)

    def forward(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod k -> inner (b mod outer) k")
        if self.pos:
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x, "inner (b mod outer) k -> b outer inner mod k", outer=self.outer, mod=self.mod,
                             b=self.batch)
        return x

class inner_att_Sparse_RA(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = My_TF_Sparse_RA(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
        self.inner_tf = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod k -> inner (b mod outer) k")
        x = self.inner_tf(x)
        x = einops.rearrange(x, "inner (b mod outer) k -> b outer inner mod k", outer=self.outer, mod=self.mod,
                             b=self.batch)
        return x

class inner_mod_att_diff_FC_cls(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward=1024, dim_proj=128):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = My_TF_diff_fc(d_model=dmodel, modalities=modalities, nhead=heads, dim_feedforward=1024)
        self.inner_tf = nn.TransformerEncoder(enc, num_layers-1)

        enc = My_TF_diff_fc(d_model=dmodel, modalities=modalities, nhead=heads, dim_feedforward=1024)
        self.inner_tf_cls = nn.TransformerEncoder(enc, 1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, dmodel))


    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod k -> mod inner (b outer) k")

        x = self.inner_tf(x)
        x = einops.rearrange(x, "mod inner (b outer) k -> b outer inner mod k", outer=self.outer, inner=self.inner, mod=self.mod, b=self.batch)
        x = torch.cat([self.cls_token.repeat( x.shape[0], x.shape[1], 1, x.shape[3], 1), x], dim=2)
        x = einops.rearrange(x, "b outer inner mod k -> mod inner (b outer) k")
        x = self.inner_tf_cls(x)
        x = einops.rearrange(x, "mod inner (b outer) k -> b outer inner mod k", outer=self.outer, b=self.batch)

        x = x[:,:,0].unsqueeze(dim=2)

        return x
class inner_att_avg_aggr(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward=1024, dim_proj = 1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
        self.inner_tf = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod k -> inner (b mod outer) k")
        if self.pos:
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_tf(x)
        x = einops.rearrange(x, "inner (b mod outer) k -> b outer inner mod k", outer=self.outer, mod=self.mod,
                             b=self.batch)
        x = x.mean(axis=2).unsqueeze(dim=2)
        return x
class inner_att_cls_aggr(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward=1024, dim_proj = 1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = My_TF(dmodel, nhead=heads, dim_feedforward=dim_feedforward, dim_proj = dim_proj)
        self.inner_tf = nn.TransformerEncoder(enc, num_layers-1)

        enc_2 = My_TF(dmodel, nhead=heads, dim_feedforward=dim_feedforward, dim_proj = dim_proj)
        self.inner_tf_2 = nn.TransformerEncoder(enc_2, 1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, dmodel))

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod ch k -> inner (b outer mod ch) k")
        if self.pos:
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_tf(x)
        x = einops.rearrange(x, "inner (b outer mod ch) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod, b=self.batch)
        x = torch.cat([self.cls_token.repeat( x.shape[0], x.shape[1], 1, x.shape[3], 1, 1), x], dim=2)
        x = einops.rearrange(x, "b outer inner mod ch k -> inner (b outer mod ch) k")
        x = self.inner_tf_2(x)
        x = x[0].unsqueeze(dim=0)
        x = einops.rearrange(x, "inner (b outer mod ch) k -> b outer inner mod ch k", outer=self.outer, mod=self.mod,
                             b=self.batch)
        return x
class inner_att_huy(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward=1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_inner = PositionalEncoder(d_model=dmodel)

        enc = TransformerEncoderLayer_Huy(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
        self.inner_tf = nn.TransformerEncoder(enc, num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[
            4]

        x = einops.rearrange(x, "b outer inner mod k -> inner (b mod outer) k")
        if self.pos:
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_tf(x)
        x = einops.rearrange(x, "inner (b mod outer) k -> b outer inner mod k", outer=self.outer, mod=self.mod,
                             b=self.batch)
        return x
class inner_att_mod(nn.Module):
        def __init__(self, dmodel, pos,  inner, outer, modalities, num_layers=1, heads=8):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel*modalities)

            enc = nn.TransformerEncoderLayer(dmodel*modalities, nhead=heads)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)
        def forward(self, x):
            x_shape = x.shape
            self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]


            x = einops.rearrange(x, "b outer inner mod k -> inner (b outer) (mod k)")
            if self.pos:
                x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = self.inner_tf(x)
            x = einops.rearrange(x, "inner (b outer) (mod k) -> b outer inner mod k", outer=self.outer, mod=self.mod,
                                 b=self.batch)
            return x
class inner_att_outer(nn.Module):
        def __init__(self, dmodel, pos,  inner, outer, modalities, num_layers=1, heads=8):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel*outer)

            enc = nn.TransformerEncoderLayer(dmodel*outer, nhead=heads)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)
        def forward(self, x):
            x_shape = x.shape
            self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]


            x = einops.rearrange(x, "b outer inner mod k -> inner (mod b) (outer k)")
            if self.pos:
                x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = self.inner_tf(x)
            x = einops.rearrange(x, "inner (mod b) (outer k) -> b outer inner mod k", outer=self.outer, mod=self.mod,
                                 b=self.batch)
            return x
class inner_att_mod_outer(nn.Module):
        def __init__(self, dmodel, pos,  inner, outer, modalities, num_layers=1, heads=8):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel*modalities*outer)

            enc = nn.TransformerEncoderLayer(dmodel*modalities*outer, nhead=heads)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)
        def forward(self, x):
            x_shape = x.shape
            self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]


            x = einops.rearrange(x, "b outer inner mod k -> inner b (outer mod k)")
            if self.pos:
                x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = self.inner_tf(x)
            x = einops.rearrange(x, " inner b (outer mod k) -> b outer inner mod k", outer=self.outer, mod=self.mod,
                                 b=self.batch)
            return x

class mod_att(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_mod = PositionalEncoder(d_model=dmodel)

        enc_mod = nn.TransformerEncoderLayer(dmodel, nhead=heads)
        self.inner_mod_tf = nn.TransformerEncoder(enc_mod,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod k -> mod (inner outer b) k")
        if self.pos:
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_mod_tf(x)
        x = einops.rearrange(x, "mod (inner outer b) k-> b outer inner mod k", outer=self.outer, inner=self.inner,
                             b=self.batch)
        return x
class mod_att_inner(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_mod = PositionalEncoder(d_model=dmodel*inner)

        enc_mod = nn.TransformerEncoderLayer(dmodel*inner, nhead=heads)
        self.inner_mod_tf = nn.TransformerEncoder(enc_mod,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod k -> mod (outer b) (inner k)")
        if self.pos:
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_mod_tf(x)
        x = einops.rearrange(x, " mod (outer b) (inner k)-> b outer inner mod k", outer=self.outer, inner=self.inner,
                             b=self.batch)
        return x
class mod_att_outer(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_mod = PositionalEncoder(d_model=dmodel*outer)

        enc_mod = nn.TransformerEncoderLayer(dmodel*outer, nhead=heads)
        self.inner_mod_tf = nn.TransformerEncoder(enc_mod,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod k -> mod (inner b) (outer k)")
        if self.pos:
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_mod_tf(x)
        x = einops.rearrange(x, "mod (inner b) (outer k)-> b outer inner mod k", outer=self.outer, inner=self.inner,
                             b=self.batch)
        return x
class mod_att_inner_outer(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_mod = PositionalEncoder(d_model=dmodel*inner*outer)

        enc_mod = nn.TransformerEncoderLayer(dmodel*inner*outer, nhead=heads)
        self.inner_mod_tf = nn.TransformerEncoder(enc_mod,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod k -> mod b (inner outer k)")
        if self.pos:
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_mod_tf(x)
        x = einops.rearrange(x, "mod b (inner outer k)-> b outer inner mod k", outer=self.outer, inner=self.inner,
                             b=self.batch)
        return x

class outer_att(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024, dim_proj= 128):
        super().__init__()

        enc = My_TF(dmodel, nhead=heads, dim_feedforward=dim_feedforward, dim_proj = dim_proj)
        self.outer_tf = nn.TransformerEncoder(enc,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod ch k -> outer (b inner mod ch) k")
        x = self.outer_tf(x)
        x = einops.rearrange(x,"outer (b inner mod ch) k-> b outer inner mod ch k", mod =self.mod, inner =self.inner, b=self.batch, ch = self.ch)
        return x

class outer_att_tf(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024, dim_proj= 128):
        super().__init__()

        enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
        self.outer_tf = nn.TransformerEncoder(enc,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> outer (b inner mod) k")
        x = self.outer_tf(x)
        x = einops.rearrange(x,"outer (b inner mod) k-> b outer inner mod k", mod =self.mod, inner =self.inner, b=self.batch)
        return x

class channel_att_RA(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, extra_attention=False, rpos=False, num_layers=1, heads=8, dim_feedforward = 1024, dim_proj= 128,  gbiased=False, ln_first=False):
        super().__init__()

        enc = My_TF_RA(dmodel, extra_attention=extra_attention, nhead=heads, rpos=rpos,  dim_feedforward=dim_feedforward, dim_proj = dim_proj, gbiased=gbiased, ln_first=ln_first)
        self.channel_tf = My_TransformerEncoder(enc,num_layers)

    def forward(self, x, extract_norm=False):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x,"b outer inner mod ch k -> ch (b outer inner mod) k")
        x = self.channel_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x,"ch (b outer inner mod) k-> b outer inner mod ch k", outer =self.outer, mod =self.mod, inner =self.inner, b=self.batch, ch=self.ch)
        return x

class outer_mod_att_RA(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, dropout=0.1, extra_attention=False, rpos=False, num_layers=1, heads=8, dim_feedforward = 1024, dim_proj= 128,  gbiased=False, ln_first=False):
        super().__init__()

        enc = My_TF_RA(dmodel, extra_attention=extra_attention, dropout=dropout, nhead=heads, rpos=rpos, dim_feedforward=dim_feedforward, dim_proj = dim_proj, gbiased=gbiased, ln_first=ln_first)
        self.outer_tf = My_TransformerEncoder(enc,num_layers)

    def forward(self, x, **kwargs):

        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x,"b outer inner mod ch k -> (outer mod ch) (b inner) k")
        x = self.outer_tf(x, **kwargs)
        x = einops.rearrange(x,"(outer mod ch) (b inner) k-> b outer inner mod ch k", outer =self.outer, mod =self.mod, inner =self.inner, b=self.batch, ch=self.ch)
        return x
class outer_mod_inner_att_RA(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, dropout,  extra_attention=False, rpos=False, num_layers=1, heads=8, dim_feedforward = 1024, dim_proj= 128,  gbiased=False, ln_first=False):
        super().__init__()

        enc = My_TF_RA(dmodel, dropout=dropout, extra_attention=extra_attention, nhead=heads, rpos=rpos, dim_feedforward=dim_feedforward, dim_proj = dim_proj, gbiased=gbiased, ln_first=ln_first)
        self.outer_tf = My_TransformerEncoder(enc,num_layers)

    def forward(self, x, extract_norm=False):

        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x,"b outer inner mod ch k -> (outer inner mod ch) b k")
        x = self.outer_tf(x, extract_norm=extract_norm)

        x = einops.rearrange(x,"(outer inner mod ch) b k-> b outer inner mod ch k", outer =self.outer, mod =self.mod, inner =self.inner, b=self.batch, ch=self.ch)
        return x
class outer_decoder_att_RA(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, extra_attention=False, rpos=False, num_layers=1, heads=8, dim_feedforward = 1024, dim_proj= 128,  gbiased=False):
        super().__init__()

        enc = My_TF_Decoder_RA(dmodel, extra_attention=extra_attention, nhead=heads, rpos=rpos, dim_feedforward=dim_feedforward, dim_proj = dim_proj, gbiased=gbiased)
        self.outer_tf = My_TransformerDecoder(enc, num_layers)

        self.outer_positional_embedding = huy_pos_outer(dmodel, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, tgt, memory, extract_norm=False):

        x_shape = memory.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        memory = einops.rearrange(memory,"b outer inner mod ch k -> (outer mod ch) (b inner) k")
        tgt = einops.rearrange(tgt,"b outer inner mod ch k -> (outer mod ch) (b inner) k")
        for i in range(self.outer):
            tgt_i = self.outer_tf(tgt=tgt, memory=memory, extract_norm=extract_norm)
            if i ==0:
                tgt = tgt_i
            else:
                tgt = torch.cat([tgt, tgt_i[-1:]],dim=0)
        tgt = einops.rearrange(tgt,"(outer mod ch) (b inner) k-> b outer inner mod ch k", outer =self.outer, mod =self.mod, inner =self.inner, b=self.batch, ch=self.ch)
        return tgt
class outer_mod_att_HPFC_RA(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, extra_attention=False, rpos=False, num_layers=1, heads=8, dim_feedforward = 1024, dim_proj= 128,  gbiased=False):
        super().__init__()

        enc = My_TF_HPFC_RA(dmodel, extra_attention=extra_attention,  modalities=modalities, nhead=heads, rpos=rpos, dim_feedforward=dim_feedforward, dim_proj = dim_proj, gbiased=gbiased)
        self.outer_tf = My_TransformerEncoder(enc,num_layers)

    def forward(self, x, extract_norm=False):

        x_shape = x[0].shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x[0] = einops.rearrange(x[0],"b outer inner mod ch k -> (outer mod ch) (b inner) k")
        x[1] = einops.rearrange(x[1],"b outer inner mod ch k -> (outer mod ch) (b inner) k")
        x = self.outer_tf(x, extract_norm=extract_norm)
        x[0] = einops.rearrange(x[0],"(outer mod ch) (b inner) k-> b outer inner mod ch k", outer =self.outer, mod =self.mod, inner =self.inner, b=self.batch, ch=self.ch)
        x[1] = einops.rearrange(x[1],"(outer mod ch) (b inner) k-> b outer inner mod ch k", outer =self.outer, mod =self.mod, inner =self.inner, b=self.batch, ch=self.ch)
        return x
class outer_mod_att_conv_RA(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, extra_attention=False, rpos=False, num_layers=1, heads=8, dim_feedforward = 1024, dim_proj= 128,  gbiased=False):
        super().__init__()

        enc = My_TF_Conv_RA(dmodel, extra_attention=extra_attention, nhead=heads, rpos=rpos, dim_feedforward=dim_feedforward, dim_proj = dim_proj, gbiased=gbiased)
        self.outer_tf = My_TransformerEncoder(enc,num_layers)

    def forward(self, x, extract_norm=False):

        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x,"b outer inner mod ch k -> (outer mod ch) (b inner) k")
        x = self.outer_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x,"(outer mod ch) (b inner) k-> b outer inner mod ch k", outer =self.outer, mod =self.mod, inner =self.inner, b=self.batch, ch=self.ch)
        return x
class outer_mod_rn_RA(nn.Module):
    def __init__(self, dmodel,  inner, outer, modalities, extra_attention=False, rpos=False, num_layers=1, bidirectional=False):
        super().__init__()

        self.outer_rn = nn.GRU(dmodel, dmodel, num_layers=num_layers, bidirectional=bidirectional)

    def forward(self, x, extract_norm=False):

        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch, self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4], x_shape[5]

        x = einops.rearrange(x,"b outer inner mod ch k -> (outer mod ch) (b inner) k")
        x, _ = self.outer_rn(x)
        x = einops.rearrange(x,"(outer mod ch) (b inner) k-> b outer inner mod ch k", outer =self.outer, mod =self.mod, inner =self.inner, b=self.batch, ch=self.ch)
        return x
class outer_mod_att_normreg_RA(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, extra_attention=False, rpos=False, num_layers=1, heads=8, dim_feedforward = 1024, dim_proj= 128,  gbiased=False):
        super().__init__()

        enc = My_TF_normreg_RA(dmodel, extra_attention=extra_attention, nhead=heads, rpos=rpos,  dim_feedforward=dim_feedforward, dim_proj = dim_proj, gbiased=gbiased)
        self.outer_tf = My_TransformerEncoder(enc,num_layers)

    def forward(self, x, extract_norm=False):

        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> (outer mod) (b inner) k")
        x = self.outer_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x,"(outer mod) (b inner) k-> b outer inner mod k", outer =self.outer, mod =self.mod, inner =self.inner, b=self.batch)
        return x
class outer_mod_intlstm(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, extra_attention=False, rpos=False, num_layers=1, heads=8, dim_feedforward = 1024, dim_proj= 128,  gbiased=False):
        super().__init__()

        enc = My_TF_LSTM(dmodel, extra_attention=extra_attention, nhead=heads, rpos=rpos,  dim_feedforward=dim_feedforward, dim_proj = dim_proj, gbiased=gbiased)
        self.outer_tf = My_TransformerEncoder(enc,num_layers)

    def forward(self, x, extract_norm=False):

        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> (outer mod) (b inner) k")
        x = self.outer_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x,"(outer mod) (b inner) k-> b outer inner mod k", outer =self.outer, mod =self.mod, inner =self.inner, b=self.batch)
        return x
class outer_mod_lstm(nn.Module):
    def __init__(self, dmodel,  inner, outer, modalities, extra_attention=False, num_layers=1, heads=8, dim_feedforward = 1024, dim_proj= 128,  gbiased=False):
        super().__init__()

        self.outer_lstm = nn.LSTM(dmodel, hidden_size = dmodel, num_layers= num_layers, bidirectional=True)

    def forward(self, x, extract_norm=False):

        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> (outer mod) (b inner) k")
        x = self.outer_lstm(x)[0]
        x = einops.rearrange(x,"(outer mod) (b inner) k-> b outer inner mod k", outer =self.outer, mod =self.mod, inner =self.inner, b=self.batch)
        return x
class outer_mod_att_fc_RA(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, extra_attention=False, rpos=False, num_layers=1, heads=8, dim_feedforward = 1024, dim_proj= 128,  gbiased=False):
        super().__init__()

        enc = My_TF_FC_RA(dmodel, extra_attention=extra_attention, nhead=heads, rpos=rpos,  modalities=modalities, dim_feedforward=dim_feedforward, dim_proj = dim_proj, gbiased=gbiased)
        self.outer_tf = My_TransformerEncoder(enc,num_layers)

    def forward(self, x, extract_norm=False):

        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> (outer mod) (b inner) k")
        x = self.outer_tf(x, extract_norm=extract_norm)
        x = einops.rearrange(x,"(outer mod) (b inner) k-> b outer inner mod k", outer =self.outer, mod =self.mod, inner =self.inner, b=self.batch)
        return x

class outer_mod_att_Norm_RA(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024, dim_proj= 128):
        super().__init__()

        enc = My_TF_Norm_RA(dmodel, nhead=heads, dim_feedforward=dim_feedforward, dim_proj = dim_proj)
        self.outer_tf = nn.TransformerEncoder(enc,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> (outer mod) (b inner) k")
        x = self.outer_tf(x)
        x = einops.rearrange(x,"(outer mod) (b inner) k-> b outer inner mod k", outer =self.outer, mod =self.mod, inner =self.inner, b=self.batch)
        return x
class outer_mod_att_Sparse_RA(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024, dim_proj= 128):
        super().__init__()

        enc = My_TF_Sparse_RA(dmodel, nhead=heads, dim_feedforward=dim_feedforward, dim_proj = dim_proj)
        self.outer_tf = nn.TransformerEncoder(enc,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> (outer mod) (b inner) k")
        x = self.outer_tf(x)
        x = einops.rearrange(x,"(outer mod) (b inner) k-> b outer inner mod k", outer =self.outer, mod =self.mod, inner =self.inner, b=self.batch)
        return x

class outer_mod_att(nn.Module):

    def __init__(self, dmodel, pos,  inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024, dim_proj= 128):
        super().__init__()

        enc = My_TF(dmodel, nhead=heads, dim_feedforward=dim_feedforward, dim_proj = dim_proj)
        self.outer_tf = nn.TransformerEncoder(enc,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> (outer inner mod) b k")
        x = self.outer_tf(x)
        x = einops.rearrange(x,"(outer inner mod) b k-> b outer inner mod k", mod =self.mod, outer =self.outer, inner =self.inner, b=self.batch)
        return x

class outer_att_huy(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=dmodel)

        enc_outer = TransformerEncoderLayer_Huy(dmodel, nhead=heads, dim_feedforward=dim_feedforward)

        self.outer_tf = nn.TransformerEncoder(enc_outer,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> outer (b inner mod) k")
        if self.pos:
            x  = self.pos_outer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.outer_tf(x)
        x = einops.rearrange(x,"outer (b inner mod) k-> b outer inner mod k", mod =self.mod, inner =self.inner, b=self.batch)
        return x
class outer_att_mod(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=dmodel*modalities)

        enc_outer = nn.TransformerEncoderLayer(dmodel*modalities, nhead=heads)
        self.outer_tf = nn.TransformerEncoder(enc_outer,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> outer (b inner) (mod k)")
        if self.pos:
            x  = self.pos_outer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.outer_tf(x)
        x = einops.rearrange(x,"outer (b inner) (mod k)-> b outer inner mod k", mod =self.mod, inner =self.inner, b=self.batch)
        return x
class outer_att_inner(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=dmodel*inner)

        enc_outer = nn.TransformerEncoderLayer(dmodel*inner, nhead=heads)
        self.outer_tf = nn.TransformerEncoder(enc_outer,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> outer (b mod) (inner k)")
        if self.pos:
            x  = self.pos_outer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.outer_tf(x)
        x = einops.rearrange(x,"outer (b mod) (inner k)-> b outer inner mod k", mod =self.mod, inner =self.inner, b=self.batch)
        return x
class outer_att_inner_mod(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=dmodel*inner*modalities)

        enc_outer = nn.TransformerEncoderLayer(dmodel*inner*modalities, nhead=heads)
        self.outer_tf = nn.TransformerEncoder(enc_outer,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> outer b (inner mod k)")
        if self.pos:
            x  = self.pos_outer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.outer_tf(x)
        x = einops.rearrange(x,"outer b (inner mod k) -> b outer inner mod k", mod =self.mod, inner =self.inner, b=self.batch)
        return x
class outer_mod_att_inner_diff_FC(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward=1024, dim_proj=128):
        super().__init__()

        enc = My_TF_diff_fc(d_model=dmodel, modalities=modalities, nhead=heads, dim_feedforward=dim_feedforward)
        self.inner_tf = nn.TransformerEncoder(enc, num_layers)


    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        x = einops.rearrange(x, "b outer inner mod k -> mod outer (b inner) k")
        x = self.inner_tf(x)
        x = einops.rearrange(x, "mod outer (b inner) k -> b outer inner mod k", b=self.batch, inner=self.inner)


        return x

class aggregation_att_outer(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Attention(dmodel)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> outer (b inner mod) k ", mod =self.mod, inner =self.inner, b=self.batch)
        w = self.mod_att(x)
        x = torch.einsum("ijk,jmi -> mjk", x, w)
        x = einops.rearrange(x," outer (b inner mod) k  -> b outer inner mod k ", b=self.batch, inner=self.inner, mod=self.mod)
        return x
class aggregation_att_inner(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Attention(dmodel*modalities)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> inner (b outer) (mod k) ", mod =self.mod, inner =self.inner, b=self.batch)
        w = self.mod_att(x)
        x = torch.einsum("ijk,jmi -> mjk", x, w)
        x = einops.rearrange(x,"inner (b outer mod) k  -> b outer inner mod k ", b=self.batch, outer=self.outer, mod=self.mod)
        return x
class aggregation_att_contx_inner(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Context_Attention(dmodel*modalities, 64)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> (b outer) inner (mod k) ", mod =self.mod, inner =self.inner, b=self.batch)

        w = self.mod_att(x)

        x = torch.einsum("ijk,im -> ik", x, w)
        x = einops.rearrange(x,"(b outer inner) (mod k)  -> b outer inner mod k ", b=self.batch, outer=self.outer, mod=self.mod, inner=1)
        return x
class aggregation_att_contx_inner_mod(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Context_Attention(dmodel, 64)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> (b outer) (inner mod) k", mod =self.mod, inner =self.inner, b=self.batch)
        w = self.mod_att(x)
        x = torch.einsum("ijk,im -> ik", x, w)
        x = einops.rearrange(x,"(b outer inner mod) k  -> b outer inner mod k ", b=self.batch, outer=self.outer, mod=1, inner=1)
        return x
class aggregation_att_contx_mod(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Context_Attention(dmodel, 64)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> (b outer inner) mod k ", mod =self.mod, inner =self.inner, b=self.batch)
        w = self.mod_att(x)
        x = torch.einsum("ijk,im -> ik", x, w)
        x = einops.rearrange(x,"(b outer inner mod)  k  -> b outer inner mod k ", b=self.batch, outer=self.outer, mod=1, inner=self.inner)
        return x
class aggregation_att_mod(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Attention(dmodel)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> mod (inner outer b) k ", mod =self.mod, inner =self.inner, b=self.batch)
        w = self.mod_att(x)
        x = torch.einsum("ijk,jmi -> mjk", x, w)
        x = einops.rearrange(x,"mod (inner outer b) k  -> b outer inner mod k -> ", inner =self.inner, b=self.batch)
        return x
class fourier_pos(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8):
        super().__init__()
        self.pos = Fourier_Sleep_PositionalEncoder(dmodel, outer, inner, modalities)

    def forward(self, x):
        return self.pos(x)
class huy_pos_inner(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8, dim_proj=0, npoints=400):
        super().__init__()
        self.pos = PositionalEncoding_AIAYN(dmodel, n_position=npoints)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod ch k -> (b outer mod ch) inner k")
        x = self.pos(x)
        x = einops.rearrange(x, "(b outer mod ch) inner k -> b outer inner mod ch k", b = self.batch, outer = self.outer, mod = self.mod, ch=self.ch)
        return x

    def forward_time(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.time, self.ch  = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        x = einops.rearrange(x, "b outer time ch k -> (b outer ch) time k")
        x = self.pos(x)
        x = einops.rearrange(x, "(b outer ch) time k-> b outer time ch k", b = self.batch, outer = self.outer, mod = self.mod, ch=self.ch)
        return x
class learnable_pos_inner(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8, dim_proj=0):
        super().__init__()
        self.pos = nn.Parameter(torch.randn([1,inner,dmodel]))

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod k -> (b outer mod) inner k")
        x = x + self.pos[:,:x.shape[1]]
        x = einops.rearrange(x, "(b outer mod) inner k -> b outer inner mod k", b = self.batch, outer = self.outer, mod = self.mod)
        return x
class huy_pos_concat_inner(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8, dim_proj=0):
        super().__init__()
        self.pos = PositionalEncoding_AIAYN(dmodel)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod k -> (b outer mod) inner k")
        x = self.pos.forward_concat(x)
        x = einops.rearrange(x, "(b outer mod) inner k -> b outer inner mod k", b = self.batch, outer = self.outer, mod = self.mod)
        return x
class huy_pos_outer(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8, dim_proj=0, npoints=400):
        super().__init__()
        self.pos = PositionalEncoding_AIAYN(dmodel, n_position=npoints)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x, "b outer inner mod ch k ->(b inner mod ch) outer k")
        x = self.pos(x)
        x = einops.rearrange(x, "(b inner mod ch) outer k -> b outer inner mod ch k", b = self.batch, inner = self.inner, mod = self.mod, ch = self.ch)
        return x

class learnable_pos_outer(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8, dim_proj=0):
        super().__init__()
        self.pos = nn.Parameter(torch.randn([1,outer,dmodel]))

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        x = einops.rearrange(x, "b outer inner mod k ->(b inner mod) outer k")
        x = x + self.pos[:,:x.shape[1]]
        x = einops.rearrange(x, "(b inner mod) outer k -> b outer inner mod k", b = self.batch, inner = self.inner, mod = self.mod)
        return x
class huy_pos_concat_outer(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, channels, num_layers=1, heads=8, dim_proj=0):
        super().__init__()
        self.pos = PositionalEncoding_AIAYN(dmodel)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        x = einops.rearrange(x, "b outer inner mod k ->(b inner mod) outer k")
        x = self.pos.forward_concat(x)
        x = einops.rearrange(x, "(b inner mod) outer k -> b outer inner mod k", b = self.batch, inner = self.inner, mod = self.mod)
        return x

class Multi_Transformer(nn.Module):

    def __init__(self, dmodel, pos, inner, outer, layers = ["inner_att", "outer_att"], modalities =1, num_layers=1, heads=8, dim_feedforward=1024, dim_proj=1024):
        super().__init__()
        self.pos = pos
        self.layers = layers
        for layer in self.layers:
            setattr(self, layer, globals()[layer](dmodel, pos, inner, outer, modalities, num_layers, heads, dim_feedforward, dim_proj))

    def forward(self,x):
        for layer in self.layers:
            this_layer = getattr(self, layer)
            x = this_layer(x)
        return x

class My_TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(My_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, src_ca: Optional[Tensor]=None, src_ca_ca: Optional[Tensor]=None, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, return_layer="last", ca_type=None, **kwargs) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        extract_norm = kwargs["extract_norm"] if "extract_norm" in kwargs else False

        output = src
        output_list = []
        for li, mod in enumerate(self.layers):
            if (src_ca is not None) and (src_ca_ca is not None):
                this_ca = src_ca[li] if ca_type=="full" else src_ca
                this_ca_ca = src_ca_ca[li] if ca_type=="full" else src_ca_ca
                output = mod(output, crossatt_src=this_ca, crossatt_src_1=this_ca_ca,  src_mask=mask, src_key_padding_mask=src_key_padding_mask, extract_norm=extract_norm)
            elif (src_ca is not None):
                this_ca = src_ca[li] if ca_type=="full" else src_ca
                output = mod(output, crossatt_src=this_ca, src_mask=mask, src_key_padding_mask=src_key_padding_mask, extract_norm=extract_norm)
            elif (src_ca_ca is not None):
                this_ca_ca = src_ca_ca[li] if ca_type=="full" else src_ca_ca
                output = mod(output, crossatt_ca_src=this_ca_ca, src_mask=mask, src_key_padding_mask=src_key_padding_mask, extract_norm=extract_norm)
            else:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, extract_norm=extract_norm)
            if return_layer != "last":
                output_list.append(output)

        if return_layer=="all":
            output = torch.cat([i.unsqueeze(dim=0) for i in output_list])
        return output

class My_TransformerEncoder_shared(nn.Module):
    def __init__(self, encoder_layers):
        super(My_TransformerEncoder_shared, self).__init__()
        for i, shared_enc in enumerate(encoder_layers):
            setattr(self, "shared_enc_{}".format(i), shared_enc)
        self.num_encoders = len(encoder_layers)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, extract_norm=False) -> Tensor:

        for enc_i in range(self.num_encoders):
            mod = getattr(self, "shared_enc_{}".format(enc_i))
            src = mod(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask, extract_norm=extract_norm)

        return src

class My_TransformerEncoder_CA_shared(nn.Module):

    def __init__(self, encoder_layers):
        super(My_TransformerEncoder_CA_shared, self).__init__()
        for i, shared_enc in enumerate(encoder_layers):
            setattr(self, "shared_enc_{}".format(i), shared_enc)
        self.num_encoders = len(encoder_layers)

    def forward(self, src: Tensor, src_ca: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, extract_norm=False) -> Tensor:

        for enc_i in range(self.num_encoders):
            mod = getattr(self, "shared_enc_{}".format(enc_i))
            src = mod(src, crossatt_src = src_ca, src_mask=mask, src_key_padding_mask=src_key_padding_mask, extract_norm=extract_norm)

        return src

class My_TransformerEncoder_CA(nn.Module):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(My_TransformerEncoder_CA, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, src_ca: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, extract_norm=False) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, crossatt_src = src_ca, src_mask=mask, src_key_padding_mask=src_key_padding_mask, extract_norm=extract_norm)

        if self.norm is not None:
            output = self.norm(output)

        return output
class My_TransformerEncoder_CA_CA(nn.Module):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(My_TransformerEncoder_CA_CA, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, src_ca_0: Tensor, src_ca_1: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, extract_norm=False) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_ca_0, src_ca_1, src_mask=mask, src_key_padding_mask=src_key_padding_mask, extract_norm=extract_norm)

        if self.norm is not None:
            output = self.norm(output)

        return output

class My_TransformerDecoder(nn.Module):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(My_TransformerDecoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, extract_norm=False) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output = output, input = memory, output_mask=tgt_mask, input_mask=memory_mask, extract_norm=extract_norm)

        if self.norm is not None:
            output = self.norm(output)

        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

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
        at = self.tanh(self.attn(x))
        at = self.softmax(einsum("bij, j -> bi",at,self.ae))
        return at


class EEG_TransferTransformer_Fusion(nn.Module):
    def __init__(self, dec, transformer_type):
        super().__init__()
        dmodel = 76
        # self.embedding = nn.Sequential(
        #     nn.Linear(dmodel, dmodel),
        #     nn.ReLU(),
        #     nn.Linear(dmodel, 128)
        # )
        dmodel = 128
        # self.latent = nn.Parameter(torch.randn(16, 8, dmodel))
        # self.cls_token = nn.Parameter(torch.randn(1, 8, dmodel))
        self.pos_inner = PositionalEncoder(d_model=dmodel, same_time_step=1)
        # transformer_type = globals()[transformer_type]
        modalities = 1
        enc = nn.TransformerEncoderLayer(dmodel, nhead=8)
        self.inner_att = nn.TransformerEncoder(enc,4)
        self.inner_att = nn.GRU(dmodel,num_layers=4)

        self.pos_outer = PositionalEncoder(d_model=dmodel*modalities, same_time_step=1)
        enc_outer = nn.TransformerEncoderLayer(dmodel*modalities, nhead=8)
        self.outer_att = nn.TransformerEncoder(enc_outer,4)
        self.outer_att = nn.GRU(dmodel, num_layers=3)
        # self.avg = nn.AvgPool2d((1,113))

    def forward(self, x):
        x_shape = x.shape
        if (len(x_shape)>3):
            x = x.flatten(start_dim=0,end_dim=1)
        # print(x.shape)
        x = self.pos_inner(x).permute(1,0,2)
        # print(x.shape)

        # xeog = self.conv_eog(xeog)
        # x = torch.cat([xeeg,xeog],dim=2)
        # x = self.pos(x.flatten(start_dim=2).permute(0, 2, 1)).permute(0, 2, 1).view(x_inner_shape)
        x = self.inner_att(x)
        # print(x.shape)
        x = x.permute(1,2,0).mean(-1) # average pooling
        x = x.view([x_shape[0], x_shape[1], -1])
        x = self.pos_outer(x.permute(1,0,2)).permute(1,0,2)
        x = self.outer_att(x)
        # print(x.shape)
        # x = self.avg(x).flatten(start_dim=1)
        # x = x.view([x_shape[0],x_shape[1], -1]).permute(1,0,2)
        # x = self.outer_att(x).permute(1,0,2).unsqueeze(dim=2)
        return x

class EEG_Embedding_Fusion_EDF(nn.Module):
    def __init__(self, dec, transformer_type):
        super().__init__()
        dmodel = 23 #76
        # self.embedding = nn.Sequential(
        #     nn.Linear(dmodel, dmodel),
        #     nn.ReLU(),
        #     nn.Linear(dmodel, 16)
        # )
        pad = 1 if dec>1 else 0
        filters = int(128/dec)
        self.embedding = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 64, kernel_size=(1, 7), stride=(1, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(64, 128, kernel_size=(1, 7), stride=(1, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((2, 2, pad, pad)),
            nn.Conv2d(128, 128, kernel_size=(dec, 5), stride=(1, 2)),
            nn.ReLU(),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(128, filters, kernel_size=(dec, 5), stride=(1, 2)),
            nn.ReLU(),
        )
        # self.embedding = nn.Sequential(
        #     nn.Conv3d(1, 64, kernel_size=(5, 1, 5), padding=(2, 0, 2), stride=(2, 1, 1)),
        #     nn.ReLU(),
        #     nn.Conv3d(64, 128, kernel_size=(5, 2, 3), padding=(0, 1, 1), stride=(2, 1, 1)),
        #     nn.ReLU(),
        #     nn.Conv3d(128, 16, kernel_size=(3, 2, 3), padding=(0, 0, 1), stride=(2, 1, 1)),
        #     nn.ReLU()
        # )
    def forward(self, x):
        # x = x.unsqueeze(dim=1)
        print(x.shape)
        x = self.embedding(x)
        print(x.shape)
        return x

class EEG_TransferGRU(nn.Module):
    def __init__(self, dec, transformer_type):
        super().__init__()
        dmodel = 128

        modalities = 1

        self.inner_gru = nn.GRU(dmodel, dmodel, num_layers=3, bidirectional=False)

        self.outer_gru = nn.GRU(dmodel*modalities, dmodel*modalities, num_layers=3, bidirectional=False)
        # self.avg = nn.AvgPool2d((1,113))

    def forward(self, x):
        x_shape = x.shape
        if (len(x_shape)>3):
            x = x.flatten(start_dim=0,end_dim=1)
        x = x.permute(1,0,2)
        x, _ = self.inner_gru(x)
        # print(x.shape)
        x = x.permute(1,2,0).mean(-1) # average pooling
        x = x.view([x_shape[0], x_shape[1], -1])
        # print(x.shape)
        x, _ = self.outer_gru(x)
        # print(x.shape)
        # x = self.avg(x).flatten(start_dim=1)
        # x = x.view([x_shape[0],x_shape[1], -1]).permute(1,0,2)
        # x = self.outer_att(x).permute(1,0,2).unsqueeze(dim=2)
        return x

class EEG_Embedding_EDF(nn.Module):
    def __init__(self, dec, transformer_type):
        super().__init__()
        dmodel = 23 #76
        # self.embedding = nn.Sequential(
        #     nn.Linear(dmodel, dmodel),
        #     nn.ReLU(),
        #     nn.Linear(dmodel, 16)
        # )
        self.embedding = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 128, kernel_size=(1, 5), stride=(1, 2)),
            nn.ReLU(),
            nn.ReflectionPad2d((2, 2, 1, 1)),
            nn.Conv2d(128, 256, kernel_size=(2, 5), stride=(1, 2)),
            nn.ReLU(),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(256, 256, kernel_size=(2, 5), stride=(1, 2)),
            nn.ReLU(),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(256, 16, kernel_size=(2, 5), stride=(1, 2)),
            nn.ReLU(),
            nn.AvgPool2d(1,13)
        )
        # self.embedding = nn.Sequential(
        #     nn.Conv3d(1, 64, kernel_size=(5, 1, 5), padding=(2, 0, 2), stride=(2, 1, 1)),
        #     nn.ReLU(),
        #     nn.Conv3d(64, 128, kernel_size=(5, 2, 3), padding=(0, 1, 1), stride=(2, 1, 1)),
        #     nn.ReLU(),
        #     nn.Conv3d(128, 16, kernel_size=(3, 2, 3), padding=(0, 0, 1), stride=(2, 1, 1)),
        #     nn.ReLU()
        # )
    def forward(self, x):
        # x = x.unsqueeze(dim=1)
        # b, outer = x.shape[0], x.shape[1]
        # x = einops.rearrange(x,"b outer ch mod inner -> (b outer) ch mod inner ")
        x = self.embedding(x)
        # x = einops.rearrange(x,"(b outer) ch mod inner  -> b outer ch mod inner ",b =b, outer= outer)
        return x

class EEG_Embedding_EDF_STFT(nn.Module):
    def __init__(self, dec, transformer_type):
        super().__init__()
        dmodel = dec
        self.embedding = nn.Sequential(
            nn.Linear(dmodel, dmodel),
            nn.ReLU(),
            nn.Linear(dmodel, 64)
        )
        # self.embedding = nn.Sequential(
        #     nn.ReflectionPad2d((2, 2, 0, 0)),
        #     nn.Conv2d(1, 128, kernel_size=(1, 5), stride=(1, 2)),
        #     nn.ReLU(),
        #     nn.ReflectionPad2d((2, 2, 1, 1)),
        #     nn.Conv2d(128, 256, kernel_size=(2, 5), stride=(1, 2)),
        #     nn.ReLU(),
        #     nn.ReflectionPad2d((2, 2, 0, 0)),
        #     nn.Conv2d(256, 256, kernel_size=(2, 5), stride=(1, 2)),
        #     nn.ReLU(),
        #     nn.ReflectionPad2d((2, 2, 0, 0)),
        #     nn.Conv2d(256, 16, kernel_size=(2, 5), stride=(1, 2)),
        #     nn.ReLU(),
        # )
        # self.embedding = nn.Sequential(
        #     nn.Conv3d(1, 64, kernel_size=(5, 1, 5), padding=(2, 0, 2), stride=(2, 1, 1)),
        #     nn.ReLU(),
        #     nn.Conv3d(64, 128, kernel_size=(5, 2, 3), padding=(0, 1, 1), stride=(2, 1, 1)),
        #     nn.ReLU(),
        #     nn.Conv3d(128, 16, kernel_size=(3, 2, 3), padding=(0, 0, 1), stride=(2, 1, 1)),
        #     nn.ReLU()
        # )
    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> b (outer inner) (mod k) ")
        x = self.embedding(x)
        x = einops.rearrange(x,"b (outer inner) (mod k) -> b outer inner mod k ", mod =1, inner =self.inner, b=self.batch)

        return x

class EEG_Embedding_EDF_STFT_1(nn.Module):
    def __init__(self, dec, transformer_type):
        super().__init__()
        dmodel = dec
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 2, 2)),
            nn.Conv2d(2, 128, kernel_size=(5, 3), stride=(1, 1)),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 2, 2)),
            nn.Conv2d(1, 128, kernel_size=(5, 3), stride=(1, 1)),
            nn.ReLU())
        self.embedding = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 2, 2)),
            nn.Conv2d(256, 256, kernel_size=(5, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 2, 2)),
            nn.Conv2d(256, 1, kernel_size=(5, 3), stride=(1, 1)),
            nn.ReLU()
        )
        # self.embedding = nn.Sequential(
        #     nn.Conv3d(1, 64, kernel_size=(5, 1, 5), padding=(2, 0, 2), stride=(2, 1, 1)),
        #     nn.ReLU(),
        #     nn.Conv3d(64, 128, kernel_size=(5, 2, 3), padding=(0, 1, 1), stride=(2, 1, 1)),
        #     nn.ReLU(),
        #     nn.Conv3d(128, 16, kernel_size=(3, 2, 3), padding=(0, 0, 1), stride=(2, 1, 1)),
        #     nn.ReLU()
        # )
    def forward(self, xeeg, xeog):

        x_shape = xeeg.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]


        xeeg = einops.rearrange(xeeg,"b outer inner mod k -> (b outer) mod k inner ")
        xeeg = self.conv1(xeeg)

        xeog = einops.rearrange(xeog,"b outer inner mod k -> (b outer) mod k inner  ")
        xeog = self.conv2(xeog)

        x = torch.cat([xeeg, xeog], dim=1)
        x = self.embedding(x)

        x = einops.rearrange(x,"(b outer) mod k inner  -> b outer inner mod k ", mod =1, outer =self.outer, b=self.batch)

        return x

class EEG_Embedding_EDF_STFT_2(nn.Module):
    def __init__(self, dec, transformer_type):
        super().__init__()
        dmodel = dec
        self.embedding = nn.Sequential(
            nn.Linear(dmodel, dmodel),
            nn.ReLU(),
            nn.Linear(dmodel, 64)
        )
        # self.embedding = nn.Sequential(
        #     nn.ReflectionPad2d((2, 2, 0, 0)),
        #     nn.Conv2d(1, 128, kernel_size=(1, 5), stride=(1, 2)),
        #     nn.ReLU(),
        #     nn.ReflectionPad2d((2, 2, 1, 1)),
        #     nn.Conv2d(128, 256, kernel_size=(2, 5), stride=(1, 2)),
        #     nn.ReLU(),
        #     nn.ReflectionPad2d((2, 2, 0, 0)),
        #     nn.Conv2d(256, 256, kernel_size=(2, 5), stride=(1, 2)),
        #     nn.ReLU(),
        #     nn.ReflectionPad2d((2, 2, 0, 0)),
        #     nn.Conv2d(256, 16, kernel_size=(2, 5), stride=(1, 2)),
        #     nn.ReLU(),
        # )
        # self.embedding = nn.Sequential(
        #     nn.Conv3d(1, 64, kernel_size=(5, 1, 5), padding=(2, 0, 2), stride=(2, 1, 1)),
        #     nn.ReLU(),
        #     nn.Conv3d(64, 128, kernel_size=(5, 2, 3), padding=(0, 1, 1), stride=(2, 1, 1)),
        #     nn.ReLU(),
        #     nn.Conv3d(128, 16, kernel_size=(3, 2, 3), padding=(0, 0, 1), stride=(2, 1, 1)),
        #     nn.ReLU()
        # )
    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod, self.ch = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]

        x = einops.rearrange(x,"b outer inner mod k -> b (outer inner) (mod k) ")
        x = self.embedding(x)
        x = einops.rearrange(x,"b (outer inner) (mod k) -> b outer inner mod k ", mod =1, inner =self.inner, b=self.batch)

        return x

class EEG_Encoder_best_SEDF_RNN_2(nn.Module):
    def __init__(self, dec, _):
        super().__init__()
        size = 64
        self.conv_0_eeg = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, size * dec, kernel_size=(1, 10), stride=(1, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((2, 2, 1, 1)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 248)),
        )
        self.conv_0_eog = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, size * dec, kernel_size=(1, 10), stride=(1, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 248)),
        )
        self.marn = MARN()
        # D_g, D_p, D_e, D_h, D_a = 150, 150, 100, 100, 100
        # self.mnc = Multilogue_Net_CategoricalModel(size * dec, size * dec, size * dec, D_g, D_p, D_e, D_h, n_classes=5,
        #                          dropout_rec=0.1, dropout=0.5)

    def forward(self, x):

        x_shape = x[0].shape
        flag_seqtoseq = False
        if len(x_shape)>4:
            flag_seqtoseq = True
            for i in range(len(x)):
                x[i] = x[i].flatten(start_dim=0, end_dim=1)


        xeeg = x[0]
        xeog = x[1]

        xeeg = self.conv_0_eeg(xeeg).flatten(start_dim=1,end_dim=2).permute(2,0,1).view([x_shape[0],x_shape[1],-1])
        xeog = self.conv_0_eog(xeog).flatten(start_dim=1,end_dim=2).permute(2,0,1).view([x_shape[0],x_shape[1],-1])
        x = self.marn(xeeg,xeog)
        # x = torch.cat([xeeg,xeog],dim=2)

        if flag_seqtoseq:
            x = x.view([x_shape[0], x_shape[1], -1])
        else:
            x = x.view([x_shape[0],-1])

        return x
class EEG_Encoder_best_SEDF_only(nn.Module):
    def __init__(self, dec, _):
        super().__init__()
        size = 64
        self.conv_0_eeg = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, size * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_0_eog = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, size * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        # self.conv_1_eeg = nn.Sequential(
        #     nn.ReflectionPad2d((2, 2, 0, 0)),
        #     nn.Conv2d(size * dec, size * dec, kernel_size=(2, 5), stride=(1, 1)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(1, 2)),
        # )
        # self.conv_1_eog = nn.Sequential(
        #     nn.ReflectionPad2d((2, 2, 0, 0)),
        #     nn.Conv2d(size * dec, size * dec, kernel_size=(2, 5), stride=(1, 1)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(1, 2)),
        # )
        # self.conv_2_eeg = nn.Sequential(
        #     nn.ReflectionPad2d((2, 2, 0, 0)),
        #     nn.Conv2d( size * dec, size * dec, kernel_size=(3, 5), stride=(1, 1)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(1, 2)),
        # )
        # self.conv_2_eog = nn.Sequential(
        #     nn.ReflectionPad2d((2, 2, 0, 0)),
        #     nn.Conv2d( size * dec, size * dec, kernel_size=(2, 5), stride=(1, 1)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(1, 2)),
        # )

        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_2 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 1, 1)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_3 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(3, 5), stride=(1, 1)),
            nn.ReLU(),
        )
        self.avg = nn.AvgPool2d((1,187))

    def forward(self, x):

        x_shape = x[0].shape
        flag_seqtoseq = False
        if len(x_shape)>4:
            flag_seqtoseq = True
            for i in range(len(x)):
                x[i] = x[i].flatten(start_dim=0, end_dim=1)


        xeeg = x[0]
        xeog = x[1]

        xeeg = self.conv_0_eeg(xeeg)
        xeog = self.conv_0_eog(xeog)

        x = self.conv_1(torch.cat([xeeg, xeog], dim=2))

        # xeeg = self.conv_1_eeg(torch.cat([xeeg[:,:,0,:].unsqueeze(dim=2), x, xeeg[:,:,1,:].unsqueeze(dim=2)], dim=2))
        # xeog = self.conv_1_eog(torch.cat([xeog, x], dim=2))

        x = self.conv_2(x)

        # xeeg = self.conv_2_eeg(torch.cat([xeeg[:, :, 0, :].unsqueeze(dim=2), x, xeeg[:, :, 1, :].unsqueeze(dim=2)], dim=2))
        # xeog = self.conv_2_eog(torch.cat([xeog, x], dim=2))

        x = self.conv_3(x)
        # print(x.shape)

        # x = torch.cat([xeeg,x,xeog],dim=2)
        x = self.avg(x)

        if flag_seqtoseq:
            x = x.view([x_shape[0], x_shape[1], -1])
        else:
            x = x.view([x_shape[0],-1])

        return x

class EEG_Encoder_best_SEDF_3(nn.Module):
    def __init__(self, dec, _):
        super().__init__()
        size = 64
        self.conv_0_eeg = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, size * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_0_eog = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, size * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_1_eeg = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_1_eog = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_2_eeg = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d( size * dec, size * dec, kernel_size=(3, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv_2_eog = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d( size * dec, size * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        self.conv_2 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(3, 5), stride=(1, 1)),
            nn.ReLU(),
        )
        self.conv_3 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(size * dec, size * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
        )
        self.avg = nn.AvgPool2d((1,187))

    def forward(self, x):

        x_shape = x[0].shape
        flag_seqtoseq = False
        if len(x_shape)>4:
            flag_seqtoseq = True
            for i in range(len(x)):
                x[i] = x[i].flatten(start_dim=0, end_dim=1)

        xeeg = x[0]
        xeog = x[1]

        xeeg = self.conv_0_eeg(xeeg)
        xeog = self.conv_0_eog(xeog)

        xeeg = self.conv_1_eeg(xeeg)
        xeog = self.conv_1_eog(xeog)

        x = self.conv_2(torch.cat([xeeg, xeog], dim=2))

        xeeg = self.conv_2_eeg(torch.cat([xeeg[:, :, 0, :].unsqueeze(dim=2), x, xeeg[:, :, 1, :].unsqueeze(dim=2)], dim=2))
        xeog = self.conv_2_eog(torch.cat([xeog, x], dim=2))

        x = self.conv_3(torch.cat([xeeg, xeog], dim=2))

        x = torch.cat([xeeg,x,xeog],dim=2)
        x = self.avg(x)
        if flag_seqtoseq:
            x = x.view([x_shape[0], x_shape[1], -1])
        else:
            x = x.view([x_shape[0],-1])
        return x

class EEG_Encoder_best_SEDF_Conv1_1(nn.Module):
    def __init__(self, dec, _):
        super().__init__()
        size = 64
        self.conv_0_eeg = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(3, size * dec, kernel_size= 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_0_eog = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2, size * dec, kernel_size= 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_1_eeg = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2* size * dec, size * dec, kernel_size= 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_1_eog = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2*size * dec, size * dec, kernel_size= 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_2_eeg = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2* size * dec, size * dec, kernel_size= 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_2_eog = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2*size * dec, size * dec, kernel_size= 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_0 = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(3, 1, kernel_size= 5),
            nn.ReLU(),
        )
        self.conv_1 = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2 * size * dec, size * dec, kernel_size= 5),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2 * size * dec, size * dec, kernel_size= 5),
            nn.ReLU(),
        )
        self.conv_3 = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2 * size * dec, 2*size * dec, kernel_size= 5),
            nn.ReLU(),
        )
        self.avg = nn.AvgPool1d((187))

    def forward(self, x):

        x_shape = x[0].shape
        flag_seqtoseq = False
        if len(x_shape)>4:
            flag_seqtoseq = True
            for i in range(len(x)):
                x[i] = x[i].flatten(start_dim=0, end_dim=1)

        xeeg = x[0].squeeze(dim=1)
        xeog = x[1].squeeze(dim=1)

        x = self.conv_0(torch.cat([xeeg, xeog], dim=1))

        xeeg = self.conv_0_eeg(torch.cat([xeeg, x], dim=1))
        xeog = self.conv_0_eog(torch.cat([xeog, x], dim=1))

        x = self.conv_1(torch.cat([xeeg, xeog], dim=1))

        xeeg = self.conv_1_eeg(torch.cat([xeeg, x], dim=1))
        xeog = self.conv_1_eog(torch.cat([xeog, x], dim=1))


        x = self.conv_2(torch.cat([xeeg, xeog], dim=1))

        xeeg = self.conv_2_eeg(torch.cat([xeeg, x], dim=1))
        xeog = self.conv_2_eog(torch.cat([xeog, x], dim=1))

        x = self.conv_3(torch.cat([xeeg, xeog], dim=1))

        x = torch.cat([xeeg,x,xeog],dim=1)

        x = self.avg(x)
        if flag_seqtoseq:
            x = x.view([x_shape[0], x_shape[1], -1])
        else:
            x = x.view([x_shape[0],-1])
        return x

class EEG_Encoder_best_SEDF_Conv1_2(nn.Module):
    def __init__(self, dec, _):
        super().__init__()
        size = 64
        self.conv_0_eeg = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2, size * dec, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_0_eog = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(1, size * dec, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_1_eeg = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2 * size * dec, size * dec, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_1_eog = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2 * size * dec, size * dec, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_2_eeg = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2 * size * dec, size * dec, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_2_eog = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2 * size * dec, size * dec, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_1 = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2 * size * dec, size * dec, kernel_size=5),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2 * size * dec, size * dec, kernel_size=5),
            nn.ReLU(),
        )
        self.conv_3 = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2 * size * dec, 2 * size * dec, kernel_size=5),
            nn.ReLU(),
        )
        self.avg = nn.AvgPool1d((187))

    def forward(self, x):

        x_shape = x[0].shape
        flag_seqtoseq = False
        if len(x_shape) > 4:
            flag_seqtoseq = True
            for i in range(len(x)):
                x[i] = x[i].flatten(start_dim=0, end_dim=1)

        xeeg = x[0].squeeze(dim=1)
        xeog = x[1].squeeze(dim=1)

        xeeg = self.conv_0_eeg(xeeg)
        xeog = self.conv_0_eog(xeog)

        x = self.conv_1(torch.cat([xeeg, xeog], dim=1))

        xeeg = self.conv_1_eeg(torch.cat([xeeg, x], dim=1))
        xeog = self.conv_1_eog(torch.cat([xeog, x], dim=1))

        x = self.conv_2(torch.cat([xeeg, xeog], dim=1))

        xeeg = self.conv_2_eeg(torch.cat([xeeg, x], dim=1))
        xeog = self.conv_2_eog(torch.cat([xeog, x], dim=1))

        x = self.conv_3(torch.cat([xeeg, xeog], dim=1))

        x = torch.cat([xeeg, x, xeog], dim=1)

        x = self.avg(x)
        if flag_seqtoseq:
            x = x.view([x_shape[0], x_shape[1], -1])
        else:
            x = x.view([x_shape[0], -1])
        return x

class EEG_Encoder_best_SEDF_Conv1_3(nn.Module):
    def __init__(self, dec, _):
        super().__init__()
        size = 64
        self.conv_0_eeg = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2, size * dec, kernel_size= 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_0_eog = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(1, size * dec, kernel_size= 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_1_eeg = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(size * dec, size * dec, kernel_size= 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_1_eog = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(size * dec, size * dec, kernel_size= 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_2_eeg = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2* size * dec, size * dec, kernel_size= 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_2_eog = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2*size * dec, size * dec, kernel_size= 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_2 = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2 * size * dec, size * dec, kernel_size= 5),
            nn.ReLU(),
        )
        self.conv_3 = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2 * size * dec, 2*size * dec, kernel_size= 5),
            nn.ReLU(),
        )
        self.avg = nn.AvgPool1d((187))

    def forward(self, x):

        x_shape = x[0].shape
        flag_seqtoseq = False
        if len(x_shape)>4:
            flag_seqtoseq = True
            for i in range(len(x)):
                x[i] = x[i].flatten(start_dim=0, end_dim=1)

        xeeg = x[0].squeeze(dim=1)
        xeog = x[1].squeeze(dim=1)

        xeeg = self.conv_0_eeg(xeeg)
        xeog = self.conv_0_eog(xeog)

        xeeg = self.conv_1_eeg(xeeg)
        xeog = self.conv_1_eog(xeog)

        x = self.conv_2(torch.cat([xeeg, xeog], dim=1))

        xeeg = self.conv_2_eeg(torch.cat([xeeg, x], dim=1))
        xeog = self.conv_2_eog(torch.cat([xeog, x], dim=1))

        x = self.conv_3(torch.cat([xeeg, xeog], dim=1))

        x = torch.cat([xeeg,x,xeog],dim=1)

        x = self.avg(x)
        if flag_seqtoseq:
            x = x.view([x_shape[0], x_shape[1], -1])
        else:
            x = x.view([x_shape[0],-1])
        return x

class EEG_Encoder_AttConv_2d(nn.Module):
    def __init__(self, dec):
        super().__init__()



        # self.pad1 = nn.ReflectionPad2d((2, 2, 1, 1))
        # self.dy_conv_0 = nn.Conv2d(1, 64 * dec, kernel_size= (1,5), stride=1)
        #     # nn.Conv2d(256 * dec, 256 * dec, kernel_size= (1,5) , stride=1, groups= 256 * dec),
        #     # nn.ReLU(),
        #     # nn.ReflectionPad2d((0, 0, 1, 1)),
        #     # nn.Conv2d(256 * dec, 512 * dec, kernel_size=(2, 1), stride=1),
        # self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=(1, 2))
        #
        # self.pad0 = nn.ReflectionPad2d((2, 2, 0, 0))
        # self.dy_conv_1 = nn.Conv2d(64 * dec, 128 * dec, kernel_size=(2, 5), stride=1)
        # self.dy_conv_2 = Dynamic_conv2d(128 * dec, 16 * dec, kernel_size=(2, 5), stride=1, K=6)
        # self.dy_conv_3 = Dynamic_conv2d(16 * dec, 16 * dec, kernel_size=(3, 3), stride=1, K=6)
        #
        #     # nn.Conv2d(512 * dec, 512 * dec, kernel_size=(1, 5), stride=1, groups= 512 * dec),
        #     # nn.ReLU(),
        #     # nn.Conv2d(128 * dec, 16 * dec, kernel_size=(2, 1), stride=1),
        #     # nn.ReLU(),
        #
        #     # AttentionConv(128 * dec, 16 * dec, kernel_size= (3, 5), stride=1, padding=[2,2,1,1]),
        #     # nn.ReLU(),
        # self.avg = nn.AvgPool2d((1, 56))

        # self.conv2 = nn.Sequential(            DynamicConv(7200*8, 3, num_heads=3),
        #                                        nn.ReLU(),
        #                                     nn.AvgPool2d((1, 56))
# )
        # import random
        # self.rands = random.sample(range(8), 8)

    def forward(self, x):

        # temp1 = copy.deepcopy(x[:,:,1,:])
        # temp3 = copy.deepcopy(x[:,:,3,:])
        # x[:, :, 1, :] = x[:,:,4,:]
        # x[:, :, 3, :] = x[:,:,6,:]
        # x[:, :, 4, :] = temp1
        # x[:, :, 6, :] = temp3
        # x = self.conv(x)
        # x = self.pad0(x)
        # x = self.dy_conv_0(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.pad1(x)
        # x = self.dy_conv_1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.pad0(x)
        # x = self.dy_conv_2(x)
        # x = self.relu(x)
        # x = self.pad0(x)
        # x = self.dy_conv_3(x)
        # x = self.relu(x)
        # # print(x.shape)
        # x = self.avg(x)
        return  self.conv(x)

class EEG_Encoder_MTMM(nn.Module):
    def __init__(self, dec):
        super().__init__()

        self.convX1 = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(2, 128 * dec, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(128 * dec, 256  * dec, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.convZ1 = nn.Sequential(
            nn.ReflectionPad1d(2),
            nn.Conv1d(1, 64 * dec, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(64 * dec, 128 * dec, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.convX2 = nn.Sequential(
            nn.MaxPool1d(4),
            nn.ReflectionPad1d(1),
            nn.Conv1d(256 * dec, 256 * dec, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(256 * dec, 128 * dec, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.convZ2 = nn.Sequential(
            nn.MaxPool1d(4),
            nn.ReflectionPad1d(1),
            nn.Conv1d(128 * dec, 128 * dec, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(128 * dec, 64 * dec, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.convX3 = nn.Sequential(
            nn.MaxPool1d(4),
            nn.ReflectionPad1d(1),
            nn.Conv1d(128 * dec, 64 * dec, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(64 * dec, 16 * dec, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.convZ3 = nn.Sequential(
            nn.MaxPool1d(4),
            nn.ReflectionPad1d(1),
            nn.Conv1d(64 * dec, 32 * dec, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(32 * dec, 16 * dec, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.avg = nn.AvgPool1d(14)
        self.mmtm1 = MMTM(128 * dec, 128 * dec, 900)
        self.mmtm2 = MMTM(64 * dec, 64 * dec, 225)
        self.mmtm3 = MMTM(16 * dec, 16 * dec, 56)

    def forward(self, x):
        x_shape = x.shape
        z = x[:,1].unsqueeze(dim=1)
        x = x[:,0].unsqueeze(dim=1)
        x = torch.cat([x,z],dim=1)

        x = self.convX1(x)
        # z = self.convX1(z)
        # x, z = self.mmtm1(x,z)

        x = self.convX2(x)
        # z = self.convX2(z)
        # x, z = self.mmtm2(x,z)

        x = self.convX3(x)
        # z = self.convX3(z)
        # x, z = self.mmtm3(x,z)
        x =  self.avg(x)
        # z =  self.avg(z)
        # x = x.view([x_shape[0],-1])

        #
        x = x.flatten(start_dim=1).unsqueeze(dim=1)
        # z = z.flatten(start_dim=1).unsqueeze(dim=1)
        # x = torch.cat([x,z],dim=1)

        return x


class EEG_Encoder_MTMM_2D(nn.Module):
    def __init__(self, dec):
        super().__init__()

        self.convX1 = nn.Sequential(
            nn.Conv2d(1, 64 * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.LayerNorm(896),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d( 64 * dec, 128 * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.LayerNorm(444),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(128 * dec, 16 * dec, kernel_size=(2, 5), stride=(1, 1)),
            nn.LayerNorm(218),
            nn.ReLU(),
            nn.AvgPool2d((1,56))
        )

        # self.convX2 = nn.Sequential(
        #     nn.ReflectionPad2d((2, 2, 0, 0)),
        #     nn.Conv2d(64 * dec, 128 * dec, kernel_size=(2, 5), stride=(1, 1), dilation=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(1, 2)),
        #     # nn.ReflectionPad2d((2, 2, 0, 0)),
        #     # nn.Conv2d(128 * dec, 16 * dec, kernel_size=(2, 5), stride=(1, 1)),
        #     # nn.ReLU()
        # )
        # self.convX3 = nn.Sequential(
        #     nn.ReflectionPad2d((2, 2, 0, 0)),
        #     nn.Conv2d(64 * dec, 128 * dec, kernel_size=(2, 5), stride=(1, 1), dilation=(2,1)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(1, 2)),
        # )
        # self.convX4 = nn.Sequential(
        #     nn.ZeroPad2d((2, 2, 0, 0)),
        #     nn.Conv2d(128 * dec, 16 * dec, kernel_size=(2, 5), stride=(1, 1)),
        #     nn.ReLU(),
        # )
        # self.max_pool = nn.MaxPool2d(kernel_size=(1, 2))
        #
        # self.convX3 = nn.Sequential(
        #     nn.ReflectionPad2d((1, 1, 2, 2)),
        #     nn.Conv2d(64 * dec, 128 * dec, kernel_size=(4, 3), stride=(1, 1)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(1, 2)),
        #     nn.ReflectionPad2d((1, 1, 1, 1)),
        #     nn.Conv2d(128 * dec, 16 * dec, kernel_size=(4, 3), stride=(1, 1)),
        #     nn.ReLU()
        # )
        # self.convX4 = nn.Sequential(
        #     nn.ReflectionPad2d((5, 5, 1, 1)),
        #     nn.Conv2d(64 * dec, 128 * dec, kernel_size=(2, 11), stride=(1, 1)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(1, 2)),
        #     nn.ReflectionPad2d((5, 5, 0, 0)),
        #     nn.Conv2d(128 * dec, 16 * dec, kernel_size=(2, 11), stride=(1, 1)),
        #     nn.ReLU()
        # )
        # self.avg = nn.AvgPool2d(kernel_size=(1,225))
        # self.mmtm1 = MMTM(64 * dec, 64 * dec, 450)
        # self.mmtm2 = MMTM(128 * dec, 128 * dec, 225)
        # self.mmtm3 = MMTM(16 * dec, 16 * dec, 56)
        # self.pos_emb = PositionalEncoder(d_model=128)
        # enc = nn.TransformerEncoderLayer(d_model=128 , nhead=16)
        # self.self_attention = nn.TransformerEncoder(encoder_layer=enc, num_layers=6)
        # #
    def forward(self, x):
        x_shape = x.shape
        # x = [x[:,i].unsqueeze(dim=1).unsqueeze(dim=1) for i in range(x_shape[1])]
        # x = [self.convX1(x[i]) for i in range(x_shape[1])]
        # x = [self.convX2(x[i]) for i in range(x_shape[1])]
        # x = torch.cat(x,dim=2)
        # z = x[:,1].unsqueeze(dim=1).unsqueeze(dim=1)
        # x = x[:,0].unsqueeze(dim=1).unsqueeze(dim=1)
        # # x = torch.cat([x,z],dim=1)
        # # x = x.unsqueeze(dim=1)
        x = self.convX1(x)
        # x2 = self.convX2(x)
        # x = self.convX3(x)
        # x = torch.cat([x2,x3],dim=2)
        # x = self.convX4(x)


        # x = self.max_pool(x)
        # z = self.convX1(z)
        # z = x[:,:,1,:]
        # x = x[:,:,0,:]
        # x, z = self.mmtm1(x,z)
        # x = torch.cat([x.unsqueeze(dim=2),z.unsqueeze(dim=2)],dim=2)

        # x = x.view([x_shape[0], x.shape[1], x_shape[1], -1])

        # x = self.convX2(x)
        # x = self.max_pool(x)

        # z = self.convX2(z)
        # z = x[:, :, 1, :]
        # x = x[:, :, 0, :]
        # x, z = self.mmtm2(x, z)
        # x = torch.cat([x.unsqueeze(dim=2), z.unsqueeze(dim=2)], dim=2)
        # x = torch.cat([x,z],dim=2)
        # x = self.convX3(x)

        # x = x.squeeze()
        # z = self.convX3(z)
        # z = z.squeeze()

        # z = x[:, :, 1, :]
        # x = x[:, :, 0, :]
        # x, z = self.mmtm3(x, z)

        # x = torch.cat([x.unsqueeze(dim=2), z.unsqueeze(dim=2)], dim=2)
        # print(x.shape)
        # print(x.shape)

        # x =  self.avg(x)
        # print(x.shape)
        # x = x.permute(0,2,1,3).flatten(start_dim=2)
        # x = self.pos_emb(x)
        # x = self.self_attention(x)
        # print(x.shape)
        # z =  self.avg(z)
        # x = x.view([x_shape[0],-1])
        #
        # x = x.flatten(start_dim=1)

        # z = z.flatten(start_dim=1).unsqueeze(dim=1)
        # x = torch.cat([x,z],dim=1)

        return x

class MMTM(nn.Module):
    def __init__(self, c1, c2, dim):
        super().__init__()
        cz = int((c1+c2)/4)
        self.common_fc = nn.Linear(c1+c2, cz)
        self.fc_a = nn.Linear(cz, c1)
        self.fc_b = nn.Linear(cz, c2)
        # self.W = nn.Parameter(torch.randn(cz, c1+c2))
        # self.W_a = nn.Parameter(torch.randn(c1, cz))
        # self.W_b = nn.Parameter(torch.randn(c2, cz))
        # self.b_a = nn.Parameter(torch.randn(c1))
        # self.b_b = nn.Parameter(torch.randn(c2))
        # self.b = nn.Parameter(torch.randn(cz))
        self.avg_1 = nn.AvgPool1d(dim)
        self.avg_2 = nn.AvgPool1d(dim)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

    def forward(self, x, z):
        sx = self.avg_1(x).squeeze()
        sz = self.avg_2(z).squeeze()
        cat = torch.torch.cat([sx,sz],dim=1)
        common_space = self.common_fc(cat)
        ea = 2*self.sigmoid(self.fc_a(common_space))
        eb = 2*self.sigmoid(self.fc_a(common_space))

        dim_diff = len(x.shape) - len(ea.shape)
        ea = ea.view(ea.shape + (1,) * dim_diff)

        dim_diff = len(z.shape) - len(eb.shape)
        eb = eb.view(eb.shape + (1,) * dim_diff)
        x, z = x*ea, z*eb

        return x, z

class EEG_Encoder_Ch_3(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.pad_1 = nn.ReflectionPad1d(2)
        self.conv1 = nn.Conv1d(1, 16*dec, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(16*dec, 64*dec, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(64*dec, 128*dec, kernel_size=3, stride=1)
        self.conv4 = nn.Conv1d(128*dec, 64*dec, kernel_size=1, stride=1)
        self.conv5 = nn.Conv1d(64*dec, 32*dec, kernel_size=3, stride=1)
        self.conv6 = nn.Conv1d(32*dec, 8*dec, kernel_size=1, stride=1)

        self.pad_2 = nn.ReflectionPad1d(1)
        # self.pad_2 = nn.ZeroPad2d((2,2,0,0))
        # self.conv2 = nn.Conv1d(10*dec, 20*dec, kernel_size=5, stride=1)

        self.maxpool_time = nn.MaxPool1d(2)
        self.avg_pool = nn.AvgPool1d(14)
        self.relu = torch.nn.ReLU()
        self.conv1_bn = nn.BatchNorm1d(1)
        self.conv2_bn = nn.BatchNorm1d(10*dec)

        # self.alpha = nn.Parameter(torch.randn(10*dec, 4),requires_grad=True)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.relu(self.conv1(self.pad_1(x)))
        x = self.relu(self.conv2(x))
        x = self.maxpool_time(x)

        x= self.relu(self.conv3(self.pad_2(x)))
        x = self.relu(self.conv4(x))
        x = self.maxpool_time(x)

        x = self.relu(self.conv5(self.pad_2(x)))
        x = self.relu(self.conv6(x))
        x = self.avg_pool(x)
        # x = self.maxpool_time(x)

        return x

class EEG_Encoder_ESeq(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.pad_1 = nn.ReflectionPad2d((5,5,1,1))
        # self.pad_1 = nn.ZeroPad2d((5,5,1,1))
        self.conv1 = nn.Conv2d(1, 10*dec, kernel_size=(2, 10), stride=(1, 1))

        self.pad_2 = nn.ReflectionPad2d((2,2,0,0))
        # self.pad_2 = nn.ZeroPad2d((2,2,0,0))
        self.conv2 = nn.Conv2d(10*dec, 20*dec, kernel_size=(1, 5), stride=(1, 1))

        self.conv3 = nn.Conv2d(20*dec, 20*dec, kernel_size=(4, 1), stride=(1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 3))
        self.maxpool_time = nn.MaxPool2d(kernel_size=(1, 3))
        self.relu = torch.nn.ReLU()
        self.conv1_bn = nn.BatchNorm2d(1)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReflectionPad2d((5, 5, 1, 1)),
            nn.Conv2d(1, 10 * dec, kernel_size=(2, 10), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 3)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(10 * dec, 20 * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(20 * dec, 20 * dec, kernel_size=(4, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
        )
        self.lstm = LSTM(400, hidden_size = 210, num_layers= 2, bidirectional=False, merge_func = lambda x : x.flatten(start_dim=1) )

    def forward(self,x):
        x_shape = x.shape
        x = self.conv(x.flatten(start_dim=0,end_dim=1)).flatten(start_dim=1)
        x = self.lstm(x.view([x_shape[0],x_shape[1],x.shape[-1]]))
        return x

class EEG_Encoder_ET(nn.Module):
    def __init__(self, dec):
        super().__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReflectionPad2d((5, 5, 1, 1)),
            nn.Conv2d(1, 10 * dec, kernel_size=(2, 10), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 3)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(10 * dec, 20 * dec, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(20 * dec, 20 * dec, kernel_size=(4, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 6)),
        )
        # self.seq = LSTM(200, hidden_size = 200, num_layers= 3, bidirectional=False, merge_func = lambda x : x.flatten(start_dim=1) )
        self.seq = Transformer_Encoder(d_model=200,nhead=8,num_layers=3)

    def forward(self,x):
        x_shape = x.shape
        x = self.conv(x.flatten(start_dim=0,end_dim=1)).flatten(start_dim=1)
        x = x.view([x_shape[0],x_shape[1],x.shape[-1]])
        x = self.seq(x)
        return x

# class EEG_Encoder_EL(nn.Module):
#     def __init__(self, dec):
#         super().__init__()
#
#         self.conv = nn.Sequential(
#             nn.BatchNorm2d(1),
#             nn.ReflectionPad2d((5, 5, 1, 1)),
#             nn.Conv2d(1, 10 * dec, kernel_size=(2, 10), stride=(1, 1)),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 3)),
#             nn.ReflectionPad2d((2, 2, 0, 0)),
#             nn.Conv2d(10 * dec, 20 * dec, kernel_size=(1, 5), stride=(1, 1)),
#             nn.ReLU(),
#             nn.Conv2d(20 * dec, 20 * dec, kernel_size=(4, 1), stride=(1, 1)),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(1, 6)),
#         )
#         self.seq = LSTM(200, hidden_size = 200, num_layers= 3, bidirectional=False, merge_func = lambda x : x.flatten(start_dim=1) )
#         # self.seq = Transformer_Encoder(d_model=200,nhead=8,num_layers=3)
#
#     def forward(self,x):
#         x_shape = x.shape
#         x = self.conv(x.flatten(start_dim=0,end_dim=1)).flatten(start_dim=1)
#         x = x.view([x_shape[0],x_shape[1],x.shape[-1]])
#         x = self.seq(x)
#         return x

class EEG_Encoder_ET_2(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.transformer = Transformer_Encoder(d_model=210,nhead=10,num_layers=3)
        self.fc = nn.Linear(720,210)

    def forward(self,x):
        x_shape = x.shape
        x = x.flatten(start_dim=2)
        x = self.fc(x)
        x = x.view([x_shape[0],x_shape[1],x.shape[-1]])
        x = self.transformer(x).flatten(start_dim=1)
        return x


class STFT_Encoder_E(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 10*dec, kernel_size=(2, 5, 5), padding=(1, 1, 2), stride=(1, 1, 1))
        self.conv2 = nn.Conv3d(10*dec, 20*dec, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1))
        self.conv3 = nn.Conv3d(20*dec, 20*dec, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=(1, 1, 1))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.maxpool_timefreq = nn.MaxPool3d(kernel_size=(4, 1, 2))
        self.relu = torch.nn.ReLU()
        self.conv1_bn = nn.BatchNorm3d(1)

    def forward(self,x):
        x1 = self.relu(self.conv1(self.conv1_bn(x)))
        x2 = self.maxpool(x1)
        x3 = self.relu(self.conv2(x2))
        x4 = self.relu(self.conv3(x3))
        x5 = self.maxpool_timefreq(x4)
        return x5

class STFT_Encoder_ET(nn.Module):
    def __init__(self, dec):

        super().__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.Conv3d(1, 10 * dec, kernel_size=(2, 5, 5), padding=(1, 1, 2), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(10 * dec, 20 * dec, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(20 * dec, 20 * dec, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(4, 2, 1))
        )

        # self.seq = LSTM(200, hidden_size = 200, num_layers= 3, bidirectional=False, merge_func = lambda x : x.flatten(start_dim=1) )
        self.seq = Transformer_Encoder(d_model=200,nhead=8,num_layers=3)

    def forward(self,x):
        x_shape = x.shape
        x = self.conv(x.flatten(start_dim=0, end_dim=1)).flatten(start_dim=1)
        x = x.view([x_shape[0], x_shape[1], x.shape[-1]])
        x = self.seq(x)
        return x

class STFT_Encoder_EL(nn.Module):
    def __init__(self, dec):

        super().__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.Conv3d(1, 10 * dec, kernel_size=(2, 5, 5), padding=(1, 1, 2), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(10 * dec, 20 * dec, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(20 * dec, 20 * dec, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(4, 2, 1))
        )

        self.seq = LSTM(200, hidden_size = 200, num_layers= 3, bidirectional=False, merge_func = lambda x : x.flatten(start_dim=1) )
        # self.seq = Transformer_Encoder(d_model=200,nhead=8,num_layers=3)

    def forward(self,x):
        x_shape = x.shape
        x = self.conv(x.flatten(start_dim=0, end_dim=1)).flatten(start_dim=1)
        x = x.view([x_shape[0], x_shape[1], x.shape[-1]])
        x = self.seq(x)
        return x

class Late_Prob_Fusion(nn.Module):
    def __init__(self, dec):
        super().__init__()

        self.conv_0 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 64 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(64 * dec, 128 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(128 * dec, 16 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.AvgPool2d((1, 56))
        )
        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 64 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(64 * dec, 128 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(128 * dec, 16 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.AvgPool2d((1, 56))
        )
        self.conv_2 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 64 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(64 * dec, 128 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(128 * dec, 16 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.AvgPool2d((1, 56))
        )
        self.conv_3 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 64 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(64 * dec, 128 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(128 * dec, 16 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.AvgPool2d((1, 56))
        )
        self.conv_4 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 64 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(64 * dec, 128 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(128 * dec, 16 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.AvgPool2d((1, 56))
        )
        self.conv_5 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 64 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(64 * dec, 128 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(128 * dec, 16 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.AvgPool2d((1, 56))
        )
        self.conv_6 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 64 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(64 * dec, 128 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(128 * dec, 16 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.AvgPool2d((1, 56))
        )

        self.conv_7 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 64 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(64 * dec, 128 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(128 * dec, 16 * dec, kernel_size=(1, 5), stride=1),
            nn.ReLU(),
            nn.AvgPool2d((1, 56))
        )

        import random
        self.rands = random.sample(range(8), 8)
        # self.rands = [7,0,1,2,3,4,5,6,7,1]
        print("Our random shuffle is:")
        print(self.rands)
    def _shuffle_channels(self,x):
        return x[:,:,self.rands,:]
    def forward(self, x):
        m = []
        m.append(self.conv_0(x[:,:,0,:].unsqueeze(dim=2)))
        m.append(self.conv_1(x[:,:,1,:].unsqueeze(dim=2)))
        m.append(self.conv_2(x[:,:,2,:].unsqueeze(dim=2)))
        m.append(self.conv_3(x[:,:,3,:].unsqueeze(dim=2)))
        m.append(self.conv_4(x[:,:,4,:].unsqueeze(dim=2)))
        m.append(self.conv_5(x[:,:,5,:].unsqueeze(dim=2)))
        m.append(self.conv_6(x[:,:,6,:].unsqueeze(dim=2)))
        m.append(self.conv_7(x[:,:,7,:].unsqueeze(dim=2)))
        x = torch.cat(m,dim=2).permute(0,2,1,3)
        # x = self._shuffle_channels(x)
        return x