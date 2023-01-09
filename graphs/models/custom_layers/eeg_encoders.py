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
    def __init__(self, dec):
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
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
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
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        if self.pos:
            x = einops.rearrange(x, "b outer inner mod k -> inner (b outer mod) k")
            x = self.pos_inner(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "inner (b outer mod) k -> mod (b outer inner) k", mod=self.mod)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "mod (b outer inner) k -> outer (b inner mod) k", outer=self.outer)
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = einops.rearrange(x, "outer (b inner mod) k -> b outer inner mod k", mod=self.mod, inner=self.inner, b = self.b)

        x = einops.rearrange(x, "b outer inner mod k -> (outer inner) mod b k")
        x0 = x[:,0,:,:]
        x1 = x[:,1,:,:]
        for i in range(self.num_layers):
            layer = getattr(self, "outter_inner_cross_att_{}".format(i))
            x0, x1 = layer(x0, x1)
        x0 = einops.rearrange(x0, "(outer inner mod) b k -> (outer inner) mod b k", outer=self.outer, inner=self.inner, mod=1)
        x1 = einops.rearrange(x1, "(outer inner mod) b k -> (outer inner) mod b k", outer=self.outer, inner=self.inner, mod=1)
        x = torch.cat([x0,x1], dim=1)
        x = einops.rearrange(x, "(outer inner) mod b k -> b outer inner mod k", outer=self.outer, mod=self.mod,
                             b=self.batch)
        return x
class inner_cross_att(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
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
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

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
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

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
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

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
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

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
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=1024)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)
        def forward(self, x):
            x_shape = x.shape
            self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

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
            self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

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
            self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

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
            self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

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
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
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
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
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
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
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
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        x = einops.rearrange(x, "b outer inner mod k -> mod b (inner outer k)")
        if self.pos:
            x = self.pos_mod(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.inner_mod_tf(x)
        x = einops.rearrange(x, "mod b (inner outer k)-> b outer inner mod k", outer=self.outer, inner=self.inner,
                             b=self.batch)
        return x
class outer_att(nn.Module):
    def __init__(self, dmodel, pos,  inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=dmodel)

        enc_outer = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=1024)
        self.outer_tf = nn.TransformerEncoder(enc_outer,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        x = einops.rearrange(x,"b outer inner mod k -> outer (b inner mod) k")
        if self.pos:
            x  = self.pos_outer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.outer_tf(x)
        x = einops.rearrange(x,"outer (b inner mod) k-> b outer inner mod k", mod =self.mod, inner =self.inner, b=self.batch)
        return x
class outer_att_mod(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=dmodel*modalities)

        enc_outer = nn.TransformerEncoderLayer(dmodel*modalities, nhead=heads)
        self.outer_tf = nn.TransformerEncoder(enc_outer,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        x = einops.rearrange(x,"b outer inner mod k -> outer (b inner) (mod k)")
        if self.pos:
            x  = self.pos_outer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.outer_tf(x)
        x = einops.rearrange(x,"outer (b inner) (mod k)-> b outer inner mod k", mod =self.mod, inner =self.inner, b=self.batch)
        return x
class outer_att_inner(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=dmodel*inner)

        enc_outer = nn.TransformerEncoderLayer(dmodel*inner, nhead=heads)
        self.outer_tf = nn.TransformerEncoder(enc_outer,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        x = einops.rearrange(x,"b outer inner mod k -> outer (b mod) (inner k)")
        if self.pos:
            x  = self.pos_outer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.outer_tf(x)
        x = einops.rearrange(x,"outer (b mod) (inner k)-> b outer inner mod k", mod =self.mod, inner =self.inner, b=self.batch)
        return x
class outer_att_inner_mod(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        if pos:
            self.pos_outer = PositionalEncoder(d_model=dmodel*inner*modalities)

        enc_outer = nn.TransformerEncoderLayer(dmodel*inner*modalities, nhead=heads)
        self.outer_tf = nn.TransformerEncoder(enc_outer,num_layers)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        x = einops.rearrange(x,"b outer inner mod k -> outer b (inner mod k)")
        if self.pos:
            x  = self.pos_outer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.outer_tf(x)
        x = einops.rearrange(x,"outer b (inner mod k) -> b outer inner mod k", mod =self.mod, inner =self.inner, b=self.batch)
        return x
class aggregation_att_outer(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Attention(dmodel)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        x = einops.rearrange(x,"b outer inner mod k -> outer (b inner mod) k ", mod =self.mod, inner =self.inner, b=self.batch)
        w = self.mod_att(x)
        x = torch.einsum("ijk,jmi -> mjk", x, w)
        x = einops.rearrange(x," outer (b inner mod) k  -> b outer inner mod k ", b=self.batch, inner=self.inner, mod=self.mod)
        return x
class aggregation_att_inner(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Attention(dmodel*modalities)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        x = einops.rearrange(x,"b outer inner mod k -> inner (b outer) (mod k) ", mod =self.mod, inner =self.inner, b=self.batch)
        w = self.mod_att(x)
        x = torch.einsum("ijk,jmi -> mjk", x, w)
        x = einops.rearrange(x,"inner (b outer mod) k  -> b outer inner mod k ", b=self.batch, outer=self.outer, mod=self.mod)
        return x
class aggregation_att_contx_inner(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Context_Attention(dmodel*modalities, 64)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        x = einops.rearrange(x,"b outer inner mod k -> (b outer) inner (mod k) ", mod =self.mod, inner =self.inner, b=self.batch)

        w = self.mod_att(x)

        x = torch.einsum("ijk,im -> ik", x, w)
        x = einops.rearrange(x,"(b outer inner) (mod k)  -> b outer inner mod k ", b=self.batch, outer=self.outer, mod=self.mod, inner=1)
        return x
class aggregation_att_contx_inner_mod(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Context_Attention(dmodel, 64)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        x = einops.rearrange(x,"b outer inner mod k -> (b outer) (inner mod) k", mod =self.mod, inner =self.inner, b=self.batch)
        w = self.mod_att(x)
        x = torch.einsum("ijk,im -> ik", x, w)
        x = einops.rearrange(x,"(b outer inner mod) k  -> b outer inner mod k ", b=self.batch, outer=self.outer, mod=1, inner=1)
        return x
class aggregation_att_contx_mod(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Context_Attention(dmodel, 64)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        x = einops.rearrange(x,"b outer inner mod k -> (b outer inner) mod k ", mod =self.mod, inner =self.inner, b=self.batch)
        w = self.mod_att(x)
        x = torch.einsum("ijk,im -> ik", x, w)
        x = einops.rearrange(x,"(b outer inner mod)  k  -> b outer inner mod k ", b=self.batch, outer=self.outer, mod=1, inner=self.inner)
        return x
class aggregation_att_mod(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.mod_att = Attention(dmodel)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        x = einops.rearrange(x,"b outer inner mod k -> mod (inner outer b) k ", mod =self.mod, inner =self.inner, b=self.batch)
        w = self.mod_att(x)
        x = torch.einsum("ijk,jmi -> mjk", x, w)
        x = einops.rearrange(x,"mod (inner outer b) k  -> b outer inner mod k -> ", inner =self.inner, b=self.batch)
        return x
class fourier_pos(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.pos = Fourier_Sleep_PositionalEncoder(dmodel, outer, inner, modalities)

    def forward(self, x):
        return self.pos(x)
class huy_pos_inner(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.pos = PositionalEncoding_AIAYN(dmodel)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        x = einops.rearrange(x, "b outer inner mod k -> (b outer mod) inner k")
        x = self.pos(x)
        x = einops.rearrange(x, "(b outer mod) inner k -> b outer inner mod k", b = self.batch, outer = self.outer, mod = self.mod)
        return x

class huy_pos_outer(nn.Module):
    def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8):
        super().__init__()
        self.pos = PositionalEncoding_AIAYN(dmodel)

    def forward(self, x):
        x_shape = x.shape
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        x = einops.rearrange(x, "b outer inner mod k ->(b inner mod) outer k")
        x = self.pos(x)
        x = einops.rearrange(x, "(b inner mod) outer k -> b outer inner mod k", b = self.batch, inner = self.inner, mod = self.mod)
        return x

class Multi_Transformer(nn.Module):

    def __init__(self, dmodel, pos, inner, outer, layers = ["inner_att", "outer_att"], modalities =1, num_layers=1, heads=8):
        super().__init__()
        self.pos = pos
        self.layers = layers
        for layer in self.layers:
            setattr(self, layer, globals()[layer](dmodel, pos, inner, outer, modalities, num_layers, heads))

    def forward(self,x):
        for layer in self.layers:
            this_layer = getattr(self, layer)
            x = this_layer(x)
        return x

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
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
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
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

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
        self.batch, self.outer, self.inner, self.mod = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
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