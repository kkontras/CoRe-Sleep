import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphs.models.attention_models.seq_base_models.mLSTM import LSTM
from graphs.models.attention_models.seq_base_models.mTransformerEncoder import Transformer_Encoder, myTransformerEncoderLayer
from graphs.models.attention_models.utils.mergeAttention import *
from graphs.models.bilstm_att import *
from graphs.models.custom_unet import *
from graphs.models.custom_layers.eeg_encoders import *
# from graphs.models.minirocket import fit, transform
# from fairseq.models.lightconv import LightConvEncoderLayer
from graphs.models.attention_models.stand_alone_att_vision import *
from graphs.models.seq2seq import seq2seq_GRU

import sys
sys.path.insert(1, '/users/sista/kkontras/Documents/Sleep_Project/Github_Project/vilbert-multi-task')
from graphs.models.attention_models.ViLBERT import MyViLBERT
from graphs.models.attention_models.MultiTransformer import Multi_Transformer_v2
from graphs.models.u2net import U2NET, U2NETP
from graphs.models.custom_layers.eeg_encoders import Multi_Transformer
from transformers import BertModel, BertConfig
from .BIOBLIP import *
from .BLIP import *

class Base_WindowFeature_Handler(nn.Module):
    def __init__(self, encs, merger, seq_handler):
        super().__init__()
        self.encs = encs
        self.merger = merger
        self.seq_handler = seq_handler

    def forward(self, x,):
        # list of views with batch x seq_length x [features]
        batch_size = x[0].shape[0]
        seq_length = x[0].shape[1]
        if x[1].shape[-1]==1:
            x[1] = x[1].squeeze(dim=len(x[1].shape)-1)
        x = [self.encs[i](xi.flatten(start_dim=0,end_dim=1)) for i, xi in enumerate(x)]

        x_cat = self.merger(x)

        #Reshape back to batch x seq_length
        new_shape = [batch_size, seq_length] + list(x_cat.shape)[1:]
        x_cat_reshaped = x_cat.view(new_shape)

        seq_out = self.seq_handler(x_cat_reshaped)

        return seq_out

class EEG_CNN_1Ch(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder"]

        d_model = 128 * 8
        fc_inner = 128
        num_classes = 2
        self.channel = channel
        print("We are processing channel {}".format(self.channel))

        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))

        self.encs = lambda x: [getattr(self,"enc_{}".format(i))(x[:,:,i,:]).flatten(start_dim=1) for i in range(x.shape[2])]

        # self.pos_emb = PositionalEncoder(d_model=128)
        # enc_0 = nn.TransformerEncoderLayer(d_model=128 , nhead=8)
        # self.self_attention = nn.TransformerEncoder(encoder_layer=enc_0, num_layers=4)

        self.fc_out = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Dropout(0.45),
            nn.Linear(d_model, fc_inner),
            nn.Sigmoid(),
            nn.Dropout(0.45),
            nn.Linear(fc_inner, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        y=[]
        for i in range():
            xi = x[0][:,:,i]
            # x_shape = x.shape
            xi = self.enc_0(xi)#.flatten(start_dim=0, end_dim=1))
            y.append(xi.unsqueeze(dim=1))
        x = torch.cat(y,dim=1).flatten(start_dim=2)
        # x = self.pos_emb(x)
        # x = self.self_attention(x)
        x = self.fc_out(x.flatten(start_dim=1))
        return x

class EEG_CNN_2Ch(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder"]

        d_model = 128 * 8
        fc_inner = 128
        num_classes = 2
        self.channel = channel
        print("We are processing channel {}".format(self.channel))

        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))

        self.encs = lambda x: [getattr(self,"enc_{}".format(i))(x[:,:,i,:]).flatten(start_dim=1) for i in range(x.shape[2])]

        self.pos_emb = PositionalEncoder(d_model=128)
        enc_0 = nn.TransformerEncoderLayer(d_model=128 , nhead=8)
        self.self_attention = nn.TransformerEncoder(encoder_layer=enc_0, num_layers=4)

        self.fc_out = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Dropout(0.45),
            nn.Linear(d_model, fc_inner),
            nn.Sigmoid(),
            nn.Dropout(0.45),
            nn.Linear(fc_inner, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        y=[]
        for i in range(8):
            xi = x[0][:,:,i].flatten(start_dim=0, end_dim=1)
            xi = self.enc_0(xi.unsqueeze(dim=1))
            y.append(xi.unsqueeze(dim=1))
        x = torch.cat(y,dim=1).flatten(start_dim=2)
        x = self.pos_emb(x)
        x = self.self_attention(x)
        x = self.fc_out(x.flatten(start_dim=1))
        return x

class EEG_CNN_Ch_Best(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder"]

        d_model = 1024
        fc_inner = 32
        num_classes = 2
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))

        self.fc_out = nn.Sequential(
                        nn.BatchNorm1d(d_model),
                        nn.Dropout(0.45),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

    def forward(self, x):
        # z = x[0][:,:,:,1:2]
        x = x[0]
        x = self.enc_0(x).flatten(start_dim=1)
        return self.fc_out(x)

class EEG_SLEEP_Neonatal(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  256#64*8
        fc_inner = 32
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        nn.BatchNorm1d(d_model),
                        nn.Dropout(0.45),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.avg = nn.AvgPool2d(kernel_size=(49, 1))

    def forward(self, x, inits= None):
        x_shape =x[0].shape
        # x = x[0]

        #Neonatal stft
        x = x[:,0,:,:,:]
        # Neonatal signal
        # x = x.flatten(start_dim=0,end_dim=1)
        # x = x.permute(0,2,1,3)

        # x = x.mean(-2)
        x = self.enc_0(x)

        # if len(x.shape) > 2:
        #     x = x.flatten(start_dim=0, end_dim=1)
        # x, h = self.gru(x)
        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_EDF78(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Dropout(0.1),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        dmodel = 128
        layers = [ "huy_pos_inner", "inner_att","aggregation_att_contx_inner", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(dmodel, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        # x = x[0]

        #Sleep EDF78
        #V3
        # x = x[0][:,:,:,0,1:,:]

        #V1
        # print(len(x))
        # print(x[0].shape)

        x = x[0][:,:,0,1:,:].unsqueeze(dim=2) #mat
        # x = x[0][:,:,:,0,1:,:] #npz
        x = einops.rearrange(x,"b outer mod f inner -> b outer inner mod f")

        # x = einops.rearrange(x,"b outer mod inner k  -> b outer inner mod k ")

        # xeog = x[1][:,:,:,:,:,1:]
        # xeeg = einops.rearrange(xeeg,"b outer f inner mod k  -> b outer inner (f mod) k ")
        # xeeg = einops.rearrange(xeeg,"b outer mod k inner -> b outer inner mod k ")
        # xeog = einops.rearrange(xeog,"b outer f inner mod k -> b outer inner (f mod) k ")
        # xeeg = self.tf1(xeeg)
        # xeog = self.tf2(xeog)

        # x = self.enc_0(xeeg,xeog)
        # xeog = self.enc_1(xeog)
        # x = torch.cat([xeeg, xeog], dim=3)

        # x = xeeg
        # x = self.tf3(x)
        x = self.tf(x)


        #Neonatal stft
        # x = x[:,0,:,:,:]
        # Neonatal signal
        # x = x.flatten(start_dim=0,end_dim=1)
        # x = x.permute(0,2,1,3)
        # x = self.enc_0(x)

        # if len(x.shape) > 2:
        #     x = x.flatten(start_dim=0, end_dim=1)
        # x, h = self.gru(x)
        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_Replicate(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        # self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        # dmodel = 128
        layers = [ "huy_pos_inner", "inner_att","aggregation_att_contx_inner", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner= 29, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        # x = x[0]

        #Sleep EDF78
        #V3
        # x = x[0][:,:,:,0,1:,:]

        #V1
        # print(len(x))
        # print(x[0].shape)

        x = x[0][:,:,:,1:,:].unsqueeze(dim=2) #mat
        # x = x[0][:,:,:,0,1:,:] #npz
        x = einops.rearrange(x,"b outer mod ch f inner -> b outer inner mod (ch f)")

        # x = einops.rearrange(x,"b outer mod inner k  -> b outer inner mod k ")

        # xeog = x[1][:,:,:,:,:,1:]
        # xeeg = einops.rearrange(xeeg,"b outer f inner mod k  -> b outer inner (f mod) k ")
        # xeeg = einops.rearrange(xeeg,"b outer mod k inner -> b outer inner mod k ")
        # xeog = einops.rearrange(xeog,"b outer f inner mod k -> b outer inner (f mod) k ")
        # xeeg = self.tf1(xeeg)
        # xeog = self.tf2(xeog)

        # x = self.enc_0(xeeg,xeog)
        # xeog = self.enc_1(xeog)
        # x = torch.cat([xeeg, xeog], dim=3)

        # x = xeeg
        # x = self.tf3(x)
        x = self.tf(x)


        #Neonatal stft
        # x = x[:,0,:,:,:]
        # Neonatal signal
        # x = x.flatten(start_dim=0,end_dim=1)
        # x = x.permute(0,2,1,3)
        # x = self.enc_0(x)

        # if len(x.shape) > 2:
        #     x = x.flatten(start_dim=0, end_dim=1)
        # x, h = self.gru(x)
        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_Replicate_AVG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        # self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        # dmodel = 128
        layers = [ "huy_pos_inner", "inner_att_avg_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner= 29, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        # x = x[0]

        #Sleep EDF78
        #V3
        # x = x[0][:,:,:,0,1:,:]

        #V1
        # print(len(x))

        x = x[0][:,:,0,1:,:].unsqueeze(dim=2).unsqueeze(dim=2) #mat
        # x = x[0][:,:,:,1:,:].unsqueeze(dim=2) #mat
        # x = x[0][:,:,:,0,1:,:] #npz
        x = einops.rearrange(x,"b outer mod ch f inner -> b outer inner mod (ch f)")

        # x = einops.rearrange(x,"b outer mod inner k  -> b outer inner mod k ")

        # xeog = x[1][:,:,:,:,:,1:]
        # xeeg = einops.rearrange(xeeg,"b outer f inner mod k  -> b outer inner (f mod) k ")
        # xeeg = einops.rearrange(xeeg,"b outer mod k inner -> b outer inner mod k ")
        # xeog = einops.rearrange(xeog,"b outer f inner mod k -> b outer inner (f mod) k ")
        # xeeg = self.tf1(xeeg)
        # xeog = self.tf2(xeog)

        # x = self.enc_0(xeeg,xeog)
        # xeog = self.enc_1(xeog)
        # x = torch.cat([xeeg, xeog], dim=3)

        # x = xeeg
        # x = self.tf3(x)
        x = self.tf(x)


        #Neonatal stft
        # x = x[:,0,:,:,:]
        # Neonatal signal
        # x = x.flatten(start_dim=0,end_dim=1)
        # x = x.permute(0,2,1,3)
        # x = self.enc_0(x)

        # if len(x.shape) > 2:
        #     x = x.flatten(start_dim=0, end_dim=1)
        # x, h = self.gru(x)
        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_Replicate_CLS(nn.Module):
    def __init__(self, encs=None, args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        # self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        # dmodel = 128
        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner= 29, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        # x = x[0]

        #Sleep EDF78
        #V3
        # x = x[0][:,:,:,0,1:,:]

        #V1
        # print(len(x))
        # print(x[0].shape)
        # print(x[0].shape)

        x = x[0][:,:,:,:,1:,:]
        # x = x[0][:,:,:,0,1:,:] #npz

        x = einops.rearrange(x,"b outer mod ch f inner -> b outer inner mod ch f")


        # x = einops.rearrange(x,"b outer mod inner k  -> b outer inner mod k ")

        # xeog = x[1][:,:,:,:,:,1:]
        # xeeg = einops.rearrange(xeeg,"b outer f inner mod k  -> b outer inner (f mod) k ")
        # xeeg = einops.rearrange(xeeg,"b outer mod k inner -> b outer inner mod k ")
        # xeog = einops.rearrange(xeog,"b outer f inner mod k -> b outer inner (f mod) k ")
        # xeeg = self.tf1(xeeg)
        # xeog = self.tf2(xeog)

        # x = self.enc_0(xeeg,xeog)
        # xeog = self.enc_1(xeog)
        # x = torch.cat([xeeg, xeog], dim=3)

        # x = xeeg
        # x = self.tf3(x)
        x = self.tf(x)


        #Neonatal stft
        # x = x[:,0,:,:,:]
        # Neonatal signal
        # x = x.flatten(start_dim=0,end_dim=1)
        # x = x.permute(0,2,1,3)
        # x = self.enc_0(x)

        # if len(x.shape) > 2:
        #     x = x.flatten(start_dim=0, end_dim=1)
        # x, h = self.gru(x)
        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_Replicate_CLS_RA(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        # self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        # dmodel = 128
        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        # self.tf = Multi_Transformer(d_model, inner= 29, outer = 21, modalities=1, heads=8,
        #                              layers = layers, num_layers=4, pos = False)

        self.inner_tf_mod0_l3_RA = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.outer_tf_mod0_l3_RA = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, inits= None):

        x = x[0][:,:,0,1:,:].unsqueeze(dim=2).unsqueeze(dim=2) #mat
        x = einops.rearrange(x,"b outer mod ch f inner -> b outer inner mod (ch f)")

        x = self.inner_positional_embedding(x)

        cls_token_eeg = self.cls_token_mod0.repeat( x.shape[0], x.shape[1], 1, x.shape[3], 1)
        x = torch.cat([cls_token_eeg, x],dim=2)
        x = self.inner_tf_mod0_l3_RA(x)
        x = x[:, :, 0].unsqueeze(dim=2)

        x = self.outer_positional_embedding(x)

        x = self.outer_tf_mod0_l3_RA(x)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_Replicate_CLS_Sparse_RA(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        # self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        # dmodel = 128
        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        # self.tf = Multi_Transformer(d_model, inner= 29, outer = 21, modalities=1, heads=8,
        #                              layers = layers, num_layers=4, pos = False)

        self.inner_tf_mod0_l3_RA = inner_att_Sparse_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.outer_tf_mod0_l3_RA = outer_mod_att_Sparse_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, inits= None):

        x = x[0][:,:,0,1:,:].unsqueeze(dim=2).unsqueeze(dim=2) #mat
        x = einops.rearrange(x,"b outer mod ch f inner -> b outer inner mod (ch f)")

        x = self.inner_positional_embedding(x)

        cls_token_eeg = self.cls_token_mod0.repeat( x.shape[0], x.shape[1], 1, x.shape[3], 1)
        x = torch.cat([cls_token_eeg, x],dim=2)
        x = self.inner_tf_mod0_l3_RA(x)
        x = x[:, :, 0].unsqueeze(dim=2)

        x = self.outer_positional_embedding(x)

        x = self.outer_tf_mod0_l3_RA(x)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_Replicate_CLS_h4_Sparse_RA(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        self.inner_tf_mod0_l0_RA = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=8)
        self.inner_tf_mod0_l1_RA = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=8)
        self.inner_tf_mod0_l2_RA = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=8)
        self.inner_tf_mod0_l3_RA = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=8)

        self.outer_tf_mod0_l0_RA = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=8)
        self.outer_tf_mod0_l1_RA = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=8)
        self.outer_tf_mod0_l2_RA = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=8)
        self.outer_tf_mod0_l3_RA = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=8)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, inits= None):

        x = x[0][:,:,0,1:,:].unsqueeze(dim=2).unsqueeze(dim=2) #mat
        x = einops.rearrange(x,"b outer mod ch f inner -> b outer inner mod (ch f)")

        x = self.inner_positional_embedding(x)

        x = self.inner_tf_mod0_l0_RA(x)
        x = self.inner_tf_mod0_l1_RA(x)
        x = self.inner_tf_mod0_l2_RA(x)

        cls_token_eeg = self.cls_token_mod0.repeat( x.shape[0], x.shape[1], 1, x.shape[3], 1)
        x = torch.cat([cls_token_eeg, x],dim=2)
        x = self.inner_tf_mod0_l3_RA(x)
        x = x[:, :, 0].unsqueeze(dim=2)

        x = self.outer_positional_embedding(x)

        x = self.outer_tf_mod0_l0_RA(x)
        x = self.outer_tf_mod0_l1_RA(x)
        x = self.outer_tf_mod0_l2_RA(x)
        x = self.outer_tf_mod0_l3_RA(x)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_Replicate_CLS_h8_Sparse_RA(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        # self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        # dmodel = 128
        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        # self.tf = Multi_Transformer(d_model, inner= 29, outer = 21, modalities=1, heads=8,
        #                              layers = layers, num_layers=4, pos = False)

        self.inner_tf_mod0_l3_RA = inner_att_Sparse_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=8)

        self.outer_tf_mod0_l3_RA = outer_mod_att_Sparse_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=8)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, inits= None):

        x = x[0][:,:,0,1:,:].unsqueeze(dim=2).unsqueeze(dim=2) #mat
        x = einops.rearrange(x,"b outer mod ch f inner -> b outer inner mod (ch f)")

        x = self.inner_positional_embedding(x)

        cls_token_eeg = self.cls_token_mod0.repeat( x.shape[0], x.shape[1], 1, x.shape[3], 1)
        x = torch.cat([cls_token_eeg, x],dim=2)
        x = self.inner_tf_mod0_l3_RA(x)
        x = x[:, :, 0].unsqueeze(dim=2)

        x = self.outer_positional_embedding(x)

        x = self.outer_tf_mod0_l3_RA(x)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_Replicate_CLS_OP(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        # self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        # dmodel = 128
        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner= 29, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, dim_feedforward=2048, dim_proj=1024, pos = False)


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        # x = x[0]

        #Sleep EDF78
        #V3
        # x = x[0][:,:,:,0,1:,:]

        #V1
        # print(len(x))
        # print(x[0].shape)

        x = x[0][:,:,:,1:,:].unsqueeze(dim=2) #mat
        # x = x[0][:,:,:,0,1:,:] #npz
        x = einops.rearrange(x,"b outer mod ch f inner -> b outer inner mod (ch f)")

        # x = einops.rearrange(x,"b outer mod inner k  -> b outer inner mod k ")

        # xeog = x[1][:,:,:,:,:,1:]
        # xeeg = einops.rearrange(xeeg,"b outer f inner mod k  -> b outer inner (f mod) k ")
        # xeeg = einops.rearrange(xeeg,"b outer mod k inner -> b outer inner mod k ")
        # xeog = einops.rearrange(xeog,"b outer f inner mod k -> b outer inner (f mod) k ")
        # xeeg = self.tf1(xeeg)
        # xeog = self.tf2(xeog)

        # x = self.enc_0(xeeg,xeog)
        # xeog = self.enc_1(xeog)
        # x = torch.cat([xeeg, xeog], dim=3)

        # x = xeeg
        # x = self.tf3(x)
        x = self.tf(x)


        #Neonatal stft
        # x = x[:,0,:,:,:]
        # Neonatal signal
        # x = x.flatten(start_dim=0,end_dim=1)
        # x = x.permute(0,2,1,3)
        # x = self.enc_0(x)

        # if len(x.shape) > 2:
        #     x = x.flatten(start_dim=0, end_dim=1)
        # x, h = self.gru(x)
        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_KD_EEG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel=0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model = 128  # 64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))

        self.fc_out_eeg = nn.Sequential(
            # nn.BatchNorm1d(d_model),
            nn.Linear(d_model, fc_inner),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fc_inner, fc_inner),
            nn.ReLU(),
            nn.Dropout(0.1),

            # nn.Dropout(0.45),
            nn.Linear(fc_inner, num_classes),
            nn.Softmax(dim=1)
        )

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.TA_token = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)


    def forward(self, x, inits=None):

        xeeg = x[0][:, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer ch f inner -> b outer inner (ch f)").unsqueeze(dim=-2)

        xeeg = self.inner_positional_embedding(xeeg)

        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        TA_token_eeg = self.TA_token.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, TA_token_eeg], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, xeeg_TA = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -1:]

        xeeg = self.outer_positional_embedding(xeeg)
        xeeg_TA = self.outer_positional_embedding(xeeg_TA)



        xeeg = self.outer_tf_mod0_l0(xeeg)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg = self.outer_tf_mod0_l3(xeeg)

        xeeg_TA = self.outer_tf_mod0_l0(xeeg_TA)
        xeeg_TA = self.outer_tf_mod0_l1(xeeg_TA)
        xeeg_TA = self.outer_tf_mod0_l2(xeeg_TA)
        xeeg_TA = self.outer_tf_mod0_l3(xeeg_TA)

        if len(xeeg.shape) > 2:
            xeeg = xeeg.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)

        if len(xeeg_TA.shape) > 2:
            xeeg_TA = xeeg_TA.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)

        xeeg = self.fc_out_eeg(xeeg)
        xeeg_TA = self.fc_out_eeg(xeeg_TA)

        return xeeg, xeeg_TA


class EEG_SleepTransformer_Neonatal(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 2
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        # nn.Linear(fc_inner, fc_inner),
                        # nn.ReLU(),
                        # nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        # self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        # dmodel = 128
        layers = [ "huy_pos_inner", "inner_mod_att","aggregation_att_contx_inner_mod", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner= 29, outer = 21, modalities=9, heads=8,
                                     layers = layers, num_layers=1, pos = False)


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        # x = x[0]

        #Sleep EDF78
        #V3
        # x = x[0][:,:,:,0,1:,:]

        #V1
        # print(len(x))
        # print(x[0].shape)

        x = x[0][:,:,:,1:,:].unsqueeze(dim=2) #mat
        # x = x[0][:,:,:,0,1:,:] #npz
        x = einops.rearrange(x,"b outer mod ch f inner -> b outer inner (mod ch) f")

        # x = einops.rearrange(x,"b outer mod inner k  -> b outer inner mod k ")

        # xeog = x[1][:,:,:,:,:,1:]
        # xeeg = einops.rearrange(xeeg,"b outer f inner mod k  -> b outer inner (f mod) k ")
        # xeeg = einops.rearrange(xeeg,"b outer mod k inner -> b outer inner mod k ")
        # xeog = einops.rearrange(xeog,"b outer f inner mod k -> b outer inner (f mod) k ")
        # xeeg = self.tf1(xeeg)
        # xeog = self.tf2(xeog)

        # x = self.enc_0(xeeg,xeog)
        # xeog = self.enc_1(xeog)
        # x = torch.cat([xeeg, xeog], dim=3)

        # x = xeeg
        # x = self.tf3(x)
        # print(x.shape)
        x = self.tf(x)


        #Neonatal stft
        # x = x[:,0,:,:,:]
        # Neonatal signal
        # x = x.flatten(start_dim=0,end_dim=1)
        # x = x.permute(0,2,1,3)
        # x = self.enc_0(x)

        # if len(x.shape) > 2:
        #     x = x.flatten(start_dim=0, end_dim=1)
        # x, h = self.gru(x)
        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_Replicate_m2(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        # self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        # dmodel = 128
        layers = [ "huy_pos_inner", "inner_att_huy","aggregation_att_contx_inner", "huy_pos_outer", "outer_att_huy"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner= 29, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        # x = x[0]

        #Sleep EDF78
        #V3
        # x = x[0][:,:,:,0,1:,:]

        #V1
        # print(len(x))
        # print(x[0].shape)

        x = x[0][:,:,:,1:,:].unsqueeze(dim=2) #mat
        # x = x[0][:,:,:,0,1:,:] #npz
        x = einops.rearrange(x,"b outer mod ch f inner -> b outer inner mod (ch f)")

        # x = einops.rearrange(x,"b outer mod inner k  -> b outer inner mod k ")

        # xeog = x[1][:,:,:,:,:,1:]
        # xeeg = einops.rearrange(xeeg,"b outer f inner mod k  -> b outer inner (f mod) k ")
        # xeeg = einops.rearrange(xeeg,"b outer mod k inner -> b outer inner mod k ")
        # xeog = einops.rearrange(xeog,"b outer f inner mod k -> b outer inner (f mod) k ")
        # xeeg = self.tf1(xeeg)
        # xeog = self.tf2(xeog)

        # x = self.enc_0(xeeg,xeog)
        # xeog = self.enc_1(xeog)
        # x = torch.cat([xeeg, xeog], dim=3)

        # x = xeeg
        # x = self.tf3(x)
        x = self.tf(x)


        #Neonatal stft
        # x = x[:,0,:,:,:]
        # Neonatal signal
        # x = x.flatten(start_dim=0,end_dim=1)
        # x = x.permute(0,2,1,3)
        # x = self.enc_0(x)

        # if len(x.shape) > 2:
        #     x = x.flatten(start_dim=0, end_dim=1)
        # x, h = self.gru(x)
        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_Contrastive(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["TF"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            elif enc == "TF":
                layers = ["huy_pos_inner", "inner_att", "aggregation_att_contx_inner", "huy_pos_outer", "outer_att"]
                tf = Multi_Transformer(d_model, inner= 29, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)
                setattr(self, "enc_{}".format(i), tf)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        # self.fc_out = nn.Sequential(
        #                 # nn.BatchNorm1d(d_model),
        #                 nn.Linear(d_model, fc_inner),
        #                 nn.ReLU(),
        #                 nn.Dropout(0.1),
        #                 nn.Linear(fc_inner, fc_inner),
        #                 nn.ReLU(),
        #                 nn.Dropout(0.1),
        #
        #                 # nn.Dropout(0.45),
        #                 nn.Linear(fc_inner, num_classes),
        #                 nn.Softmax(dim=1)
        #             )
        # self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        # dmodel = 128
        # print("Our layers are: \n {}".format(layers))
        # self.tf = Multi_Transformer(d_model, inner= 29, outer = 21, modalities=1, heads=8,
        #                              layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        # x = x[0]

        #Sleep EDF78
        #V3
        # x = x[0][:,:,:,0,1:,:]

        #V1
        # print(len(x))
        # print(x[0].shape)

        x = x[0][:,:,:,1:,:].unsqueeze(dim=2) #mat
        x = einops.rearrange(x,"b outer mod ch f inner -> b outer inner mod (ch f)")

        x = self.enc_0(x)

        return x

class EEG_SLEEP_Contrastive_FC(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["TF"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            elif enc == "TF":
                layers = ["huy_pos_inner", "inner_att", "aggregation_att_contx_inner", "huy_pos_outer", "outer_att"]
                tf = Multi_Transformer(d_model, inner= 29, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)
                setattr(self, "enc_{}".format(i), tf)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        # self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        # dmodel = 128
        # print("Our layers are: \n {}".format(layers))
        # self.tf = Multi_Transformer(d_model, inner= 29, outer = 21, modalities=1, heads=8,
        #                              layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        # x = x[0]

        #Sleep EDF78
        #V3
        # x = x[0][:,:,:,0,1:,:]

        #V1
        # print(len(x))
        # print(x[0].shape)

        x = x[0][:,:,:,1:,:].unsqueeze(dim=2) #mat
        x = einops.rearrange(x,"b outer mod ch f inner -> b outer inner mod (ch f)")

        x = self.enc_0(x)
        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiChannel(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128*2#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Dropout(0.1),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        layers = [ "huy_pos_inner", "inner_att","aggregation_att_contx_inner", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        # x = x[0]

        #Sleep EDF78
        #V3
        # x = x[0][:,:,:,0,1:,:]

        #V1
        # print(len(x))
        # print(x[0].shape)

        x = x[0][:,:,:,1:,:] #mat
        # x = x[0][:,:,:,0,1:,:] #npz
        x = einops.rearrange(x,"b outer ch f inner -> b outer inner (ch f)").unsqueeze(dim=-2)

        x = self.tf(x)


        #Neonatal stft
        # x = x[:,0,:,:,:]
        # Neonatal signal
        # x = x.flatten(start_dim=0,end_dim=1)
        # x = x.permute(0,2,1,3)
        # x = self.enc_0(x)

        # if len(x.shape) > 2:
        #     x = x.flatten(start_dim=0, end_dim=1)
        # x, h = self.gru(x)
        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiChannelMultiModal_concat(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128*5#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Dropout(0.1),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        layers = [ "huy_pos_inner", "inner_att","aggregation_att_contx_inner", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        # x = x[0]

        #Sleep EDF78
        #V3
        # x = x[0][:,:,:,0,1:,:]

        #V1
        # print(len(x))
        # print(x[0].shape)

        xeeg = x[0][:,:,:,1:,:] #mat
        xemg = x[1][:,:,:,1:,:] #mat
        xeog = x[2][:,:,:,1:,:] #mat
        x = torch.cat([xeeg, xemg, xeog], dim=2)
        # x = x[0][:,:,:,0,1:,:] #npz
        x = einops.rearrange(x,"b outer ch f inner -> b outer inner (ch f)").unsqueeze(dim=-2)

        x = self.tf(x)


        #Neonatal stft
        # x = x[:,0,:,:,:]
        # Neonatal signal
        # x = x.flatten(start_dim=0,end_dim=1)
        # x = x.permute(0,2,1,3)
        # x = self.enc_0(x)

        # if len(x.shape) > 2:
        #     x = x.flatten(start_dim=0, end_dim=1)
        # x, h = self.gru(x)
        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiChannelMultiModal_merged(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128*2#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Dropout(0.1),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        layers = [ "huy_pos_inner", "inner_att","aggregation_att_contx_inner", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        # x = x[0]

        #Sleep EDF78
        #V3
        # x = x[0][:,:,:,0,1:,:]

        #V1
        # print(len(x))
        # print(x[0].shape)

        xeeg = x[0][:,:,:,1:,:] #mat
        # xemg = x[2][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat
        x = torch.cat([xeeg, xeog], dim=-1)
        # x = x[0][:,:,:,0,1:,:] #npz
        x = einops.rearrange(x,"b outer ch f inner -> b outer inner (ch f)").unsqueeze(dim=-2)

        x = self.tf(x)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiChannelMultiModal_merged_with_diff_FC(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel=0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model = 128  # 64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))

        self.fc_out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, fc_inner),
            nn.ReLU(),
            nn.Linear(fc_inner, num_classes),
            nn.Softmax(dim=1)
        )
        self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        layers = ["huy_pos_inner", "inner_mod_att_ch_diff_FC", "aggregation_att_contx_inner_ch", "huy_pos_outer", "outer_att_ch"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer_v2(d_model * 2, inner=30, outer=21, modalities=2, channels=2, heads=8,
                                        layers=layers, num_layers=4, pos=False)

    def forward(self, x, inits=None):

        xeeg = x[0][:, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer ch f inner -> b outer inner ch f").unsqueeze(dim=-3)
        xeog = einops.rearrange(xeog, "b outer ch f inner -> b outer inner ch f").unsqueeze(dim=-3)

        x = torch.cat([xeeg, xeog], dim=-3)

        x = self.tf(x)

        if len(x.shape) > 2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiChannelMultiModal_merged_io_with_diff_FC(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel=0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model = 128  # 64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))

        self.fc_out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, fc_inner),
            nn.ReLU(),
            nn.Linear(fc_inner, num_classes),
            nn.Softmax(dim=1)
        )
        self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        layers = ["huy_pos_inner", "inner_mod_att_ch_diff_FC", "aggregation_att_contx_inner_ch", "huy_pos_outer", "outer_mod_att_inner_ch_diff_FC"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer_v2(d_model * 2, inner=30, outer=21, modalities=2, channels=2, heads=8,
                                        layers=layers, num_layers=4, pos=False)

    def forward(self, x, inits=None):

        xeeg = x[0][:, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer ch f inner -> b outer inner ch f").unsqueeze(dim=-3)
        xeog = einops.rearrange(xeog, "b outer ch f inner -> b outer inner ch f").unsqueeze(dim=-3)

        x = torch.cat([xeeg, xeog], dim=-3)

        x = self.tf(x)

        if len(x.shape) > 2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_late(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Dropout(0.1),
                        nn.Linear(d_model*5, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        layers = [ "huy_pos_inner", "inner_att","aggregation_att_contx_inner", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf_eeg = Multi_Transformer(d_model*2, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)

        self.tf_emg = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)

        self.tf_eog = Multi_Transformer(d_model*2, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xemg = x[2][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

                # x = x[0][:,:,:,0,1:,:] #npz
        xeeg = einops.rearrange(xeeg,"b outer ch f inner -> b outer inner (ch f)").unsqueeze(dim=-2)
        xemg = einops.rearrange(xemg,"b outer ch f inner -> b outer inner (ch f)").unsqueeze(dim=-2)
        xeog = einops.rearrange(xeog,"b outer ch f inner -> b outer inner (ch f)").unsqueeze(dim=-2)

        xeeg = self.tf_eeg(xeeg)
        xeog = self.tf_eog(xeog)
        xemg = self.tf_emg(xemg)

        x = torch.cat([xeeg, xeog, xemg], dim=-1)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SHHS_CNN_SingleMod(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder"]

        d_model = 128
        fc_inner = 1024
        num_classes = 5

        self.conv = nn.Sequential(
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(1, 64, kernel_size= 5, stride=2),
            nn.MaxPool1d(kernel_size=2),
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(128, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d((1, 23))  # 56

        )

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

    def forward(self, x):
        # z = x[0][:,:,:,1:2]
        x = x[0].unsqueeze(dim=2)
        x = einops.rearrange(x, "b outer ch inner -> (b outer) ch inner")
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc_out(x)

class EEG_SHHS_U2net_SingleMod(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder"]

        d_model = 100
        fc_inner = 1024
        num_classes = 5

        self.u2net = U2NETP()
        self.avg = nn.AvgPool2d(kernel_size=(1,150))
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

    def forward(self, x):
        # z = x[0][:,:,:,1:2]
        x = x[0].unsqueeze(dim=2).unsqueeze(dim=2)
        x = einops.rearrange(x, "b outer mod ch inner -> (b outer) mod ch inner")
        x = self.u2net(x)

        x = torch.cat(x,dim=1)
        x = self.avg(x)

        x = x.flatten(start_dim=1)

        return self.fc_out(x)

class EEG_SHHS_CNN_late_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder"]

        d_model = 128
        fc_inner = 1024
        num_classes = 5

        self.conv_eeg = nn.Sequential(
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(1, 64, kernel_size= 5, stride=2),
            nn.MaxPool1d(kernel_size=2),
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(128, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d((1, 23))  # 56
        )
        self.conv_eog = nn.Sequential(
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(1, 64, kernel_size= 5, stride=2),
            nn.MaxPool1d(kernel_size=2),
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(128, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d((1, 23))  # 56
        )

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

    def forward(self, x):
        xeeg = x[0].unsqueeze(dim=2)
        xeog = x[1].unsqueeze(dim=2)

        xeeg = einops.rearrange(xeeg, "b outer ch inner -> (b outer) ch inner")
        xeog = einops.rearrange(xeog, "b outer ch inner -> (b outer) ch inner")

        xeeg = self.conv_eeg(xeeg)
        xeog = self.conv_eog(xeog)

        x = torch.cat([xeeg,xeog], dim=1)

        x = x.flatten(start_dim=1)
        return self.fc_out(x)

class EEG_SHHS_CNN_late_late_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder"]

        d_model = 128
        fc_inner = 1024
        num_classes = 5

        self.conv_eeg = nn.Sequential(
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(1, 64, kernel_size= 5, stride=2),
            nn.MaxPool1d(kernel_size=2),
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(128, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d((1, 23))  # 56
        )
        self.conv_eog = nn.Sequential(
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(1, 64, kernel_size= 5, stride=2),
            nn.MaxPool1d(kernel_size=2),
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(128, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d((1, 23))  # 56
        )

        self.fc_out_eeg = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.fc_out_eog = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

    def forward(self, x):
        xeeg = x[0].unsqueeze(dim=2)
        xeog = x[1].unsqueeze(dim=2)

        xeeg = einops.rearrange(xeeg, "b outer ch inner -> (b outer) ch inner")
        xeog = einops.rearrange(xeog, "b outer ch inner -> (b outer) ch inner")

        xeeg = self.conv_eeg(xeeg).flatten(start_dim=1)
        xeog = self.conv_eog(xeog).flatten(start_dim=1)

        xeeg = self.fc_out_eeg(xeeg).unsqueeze(dim=1)
        xeog = self.fc_out_eog(xeog).unsqueeze(dim=1)

        x = torch.cat([xeeg,xeog], dim=1).mean(dim=1).squeeze()

        return x

class EEG_SHHS_CNN_mid_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder"]

        d_model = 128
        fc_inner = 1024
        num_classes = 5

        self.conv_eeg = nn.Sequential(
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(1, 64, kernel_size= 5, stride=2),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv_eog = nn.Sequential(
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(1, 64, kernel_size= 5, stride=2),
            nn.MaxPool1d(kernel_size=2),
        )

        self.conv = nn.Sequential(
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(128, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.ReflectionPad1d((2, 2)),
            nn.Conv1d(256, 32, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d((1, 23))  # 56
        )

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

    def forward(self, x):
        xeeg = x[0].unsqueeze(dim=2)
        xeog = x[1].unsqueeze(dim=2)

        xeeg = einops.rearrange(xeeg, "b outer ch inner -> (b outer) ch inner")
        xeog = einops.rearrange(xeog, "b outer ch inner -> (b outer) ch inner")

        xeeg = self.conv_eeg(xeeg)
        xeog = self.conv_eog(xeog)

        x = torch.cat([xeeg,xeog], dim=1)

        x = self.conv(x)

        x = x.flatten(start_dim=1)
        return self.fc_out(x)

class EEG_SHHS_CNN_early_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder"]

        d_model = 128
        fc_inner = 1024
        num_classes = 5

        self.conv = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 1, 0)),
            nn.Conv2d(1, 64, kernel_size= (2, 5), stride=(1,2)),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(64, 256, kernel_size=(2, 5), stride=(1,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(256, 32, kernel_size=(1, 5)),
            nn.ReLU(),
            nn.AvgPool2d((1, 23))  # 56

        )

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

    def forward(self, x):
        xeeg = x[0].unsqueeze(dim=2).unsqueeze(dim=2)
        xeog = x[1].unsqueeze(dim=2).unsqueeze(dim=2)

        xeeg = einops.rearrange(xeeg, "b outer ch mod inner -> (b outer) ch mod inner")
        xeog = einops.rearrange(xeog, "b outer ch mod inner -> (b outer) ch mod inner")

        x = torch.cat([xeeg,xeog], dim=2)

        x = self.conv(x)

        x = x.flatten(start_dim=1)
        return self.fc_out(x)

class EEG_SHHS_CNN_mid_shared_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder"]

        d_model = 128
        fc_inner = 1024
        num_classes = 5

        self.conv = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 64, kernel_size= (1, 5), stride=(1,2)),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(64, 256, kernel_size=(2, 5), stride=(1,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(256, 32, kernel_size=(1, 5)),
            nn.ReLU(),
            nn.AvgPool2d((1, 23))  # 56

        )

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

    def forward(self, x):

        xeeg = x[0].unsqueeze(dim=2).unsqueeze(dim=2)
        xeog = x[1].unsqueeze(dim=2).unsqueeze(dim=2)

        xeeg = einops.rearrange(xeeg, "b outer ch mod inner -> (b outer) ch mod inner")
        xeog = einops.rearrange(xeog, "b outer ch mod inner -> (b outer) ch mod inner")

        x = torch.cat([xeeg,xeog], dim=2)

        x = self.conv(x)

        x = x.flatten(start_dim=1)
        return self.fc_out(x)

class EEG_SLEEP_MultiModal_late_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        self.tf_eeg = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)

        self.tf_eog = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.tf_eeg(xeeg)
        xeog = self.tf_eog(xeog)

        x = torch.cat([xeeg, xeog], dim=-1)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_late_l1_RA_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out =  nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )



        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)
        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)


        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding_0 = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding_1 = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        # Inner layer 0

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, 1, 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding_0(xeeg)
        xeog = self.outer_positional_embedding_1(xeog)

        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeog = self.outer_tf_mod1_l3(xeog)

        x = torch.cat([xeeg, xeog], dim=-1)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)

        x = self.fc_out(x)

        return x

class EEG_SLEEP_MultiModal_late_EEG_EOG_EMG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*3, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf_eeg = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)

        self.tf_eog = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)

        self.tf_emg = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat
        xemg = x[2][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")
        xemg = einops.rearrange(xemg,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.tf_eeg(xeeg)
        xeog = self.tf_eog(xeog)
        xemg = self.tf_eog(xemg)

        x = torch.cat([xeeg, xeog, xemg], dim=-1)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_late_late_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))

        self.fc_out_eeg =  nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )


        self.fc_out_eog =  nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )


        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf_eeg = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)

        self.tf_eog = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

                # x = x[0][:,:,:,0,1:,:] #npz
        xeeg = einops.rearrange(xeeg,"b outer ch f inner -> b outer inner (ch f)").unsqueeze(dim=-2)
        xeog = einops.rearrange(xeog,"b outer ch f inner -> b outer inner (ch f)").unsqueeze(dim=-2)

        xeeg = self.tf_eeg(xeeg)
        xeog = self.tf_eog(xeog)

        # x = torch.cat([xeeg, xeog, xemg], dim=-1)

        if len(xeeg.shape)>2:
            xeeg = xeeg.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        if len(xeog.shape)>2:
            xeog = xeog.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)

        xeeg = self.fc_out_eeg(xeeg).unsqueeze(dim=-1)
        xeog = self.fc_out_eog(xeog).unsqueeze(dim=-1)

        x = torch.cat([xeeg, xeog], dim=-1).mean(dim=-1).squeeze()


        return x

class EEG_SLEEP_MultiModal_late_late_l1_RA_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out_eeg  = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )


        self.fc_out_eog =  nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )


        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)
        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)


        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding_0 = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding_1 = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        # Inner layer 0

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, 1, 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)
        xeog = self.inner_tf_mod0_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding_0(xeeg)
        xeog = self.outer_positional_embedding_1(xeog)

        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeog = self.outer_tf_mod1_l3(xeog)

        if len(xeog.shape)>2:
            xeog = xeog.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)

        if len(xeeg.shape)>2:
            xeeg = xeeg.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)

        xeeg = self.fc_out_eeg(xeeg)
        xeog = self.fc_out_eog(xeog)

        x = torch.cat([xeeg.unsqueeze(dim=-1), xeog.unsqueeze(dim=-1)], dim=-1)
        x=x.mean(dim=-1)

        return x

class EEG_SLEEP_MultiModal_late_late_EEG_EOG_EMG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))

        self.fc_out_eeg = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Dropout(0.1),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        self.fc_out_eog = nn.Sequential(
            # nn.BatchNorm1d(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, fc_inner),
            nn.ReLU(),
            # nn.Dropout(0.45),
            nn.Linear(fc_inner, num_classes),
            nn.Softmax(dim=1)
        )

        self.fc_out_emg = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Dropout(0.1),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )


        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf_eeg = Multi_Transformer(d_model*2, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)

        self.tf_emg = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)

        self.tf_eog = Multi_Transformer(d_model*2, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat
        xemg = x[2][:,:,:,1:,:] #mat

                # x = x[0][:,:,:,0,1:,:] #npz
        xeeg = einops.rearrange(xeeg,"b outer ch f inner -> b outer inner (ch f)").unsqueeze(dim=-2)
        xeog = einops.rearrange(xeog,"b outer ch f inner -> b outer inner (ch f)").unsqueeze(dim=-2)
        xemg = einops.rearrange(xemg,"b outer ch f inner -> b outer inner (ch f)").unsqueeze(dim=-2)

        xeeg = self.tf_eeg(xeeg)
        xeog = self.tf_eog(xeog)
        xemg = self.tf_emg(xemg)

        # x = torch.cat([xeeg, xeog, xemg], dim=-1)

        if len(xeeg.shape)>2:
            xeeg = xeeg.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        if len(xeog.shape)>2:
            xeog = xeog.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        if len(xemg.shape)>2:
            xemg = xemg.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)

        xeeg = self.fc_out_eeg(xeeg).unsqueeze(dim=-1)
        xeog = self.fc_out_eog(xeog).unsqueeze(dim=-1)
        xemg = self.fc_out_emg(xemg).unsqueeze(dim=-1)

        x = torch.cat([xeeg, xeog, xemg], dim=-1).mean(dim=-1).squeeze()

        return x

class EEG_SLEEP_MultiModal_merged_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]
        encoder_filters = 2
        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None, extract_norm=False):
        xeeg = x[0][:,:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog,"b outer mod ch f inner -> b outer inner mod ch f")

        x = torch.cat([xeeg, xeog], dim=3)

        x = self.tf(x)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_merged_RA_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        x = torch.cat([xeeg, xeog], dim=3)

        x = self.inner_positional_embedding_0(x)


        x = self.inner_tf_mod0_l0(x)
        x = self.inner_tf_mod0_l1(x)
        x = self.inner_tf_mod0_l2(x)

        cls_token_eeg = self.cls_token_mod0.repeat( x.shape[0], x.shape[1], 1, x.shape[3], 1)
        x = torch.cat([cls_token_eeg, x], dim=2)
        x = self.inner_tf_mod0_l3(x)
        x = x[:, :, 0].unsqueeze(dim=2)

        x = self.outer_positional_embedding(x)

        x = self.outer_tf_mod0_l0(x)
        x = self.outer_tf_mod0_l1(x)
        x = self.outer_tf_mod0_l2(x)
        x = self.outer_tf_mod0_l3(x)


        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_merged_l1_RA_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)


        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(128, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(128, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding_0 = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding_1 = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        x = torch.cat([xeeg, xeog], dim=2)

        # Inner layer 0

        cls_token_eeg = self.cls_token_mod0.repeat( x.shape[0], x.shape[1], 1, 1, 1)
        cls_token_eog = self.cls_token_mod1.repeat( x.shape[0], x.shape[1], 1, 1, 1)
        x = torch.cat([cls_token_eeg, x, cls_token_eog], dim=2)
        x = self.inner_tf_mod0_l3(x)
        xeeg = x[:, :, 0].unsqueeze(dim=2)
        xeog = x[:, :, -1].unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding_0(xeeg)
        xeog = self.outer_positional_embedding_1(xeog)

        outer= xeeg.shape[1]

        x = torch.cat([xeeg, xeog], dim=1)

        x = self.outer_tf_mod0_l3(x)

        x = einops.rearrange(x, "b (outer mod) inner m f -> (b outer) (inner mod m f) ", outer=outer, mod=2 )
        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x


class EEG_SLEEP_MultiModal_merged_EEG_EOG_EMG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*3, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat
        xemg = x[2][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")
        xemg = einops.rearrange(xemg,"b outer mod f inner -> b outer inner mod f")

        x = torch.cat([xeeg, xeog, xemg], dim=3)

        x = self.tf(x)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_concat_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]

        self.tf = Multi_Transformer(d_model*2, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        x = torch.cat([xeeg, xeog], dim=-1)

        x = self.tf(x)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_concat_v2_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128*2#64*8
        fc_inner = 1024
        num_classes = 5

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)


        self.outer_tf_mod0_l0 = outer_att_tf(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att_tf(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att_tf(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att_tf(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding_0 = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)


    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        x = torch.cat([xeeg, xeog], dim=-1)

        x = self.inner_positional_embedding_0(x)

        x = self.inner_tf_mod0_l0(x)
        x = self.inner_tf_mod0_l1(x)
        x = self.inner_tf_mod0_l2(x)

        # Inner layer 3
        cls_token = self.cls_token_mod0.repeat( x.shape[0], x.shape[1], 1, 1, 1)
        x = torch.cat([cls_token, x], dim=2)
        x = self.inner_tf_mod0_l3(x)
        x = x[:, :, 0].unsqueeze(dim=2)

        x = self.outer_positional_embedding_0(x)

        x = self.outer_tf_mod0_l0(x)
        x = self.outer_tf_mod0_l1(x)
        x = self.outer_tf_mod0_l2(x)
        x = self.outer_tf_mod0_l3(x)


        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_concat_l1_RA_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128*2#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)


        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(128, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(128, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        x = torch.cat([xeeg, xeog], dim=-1)


        # Inner layer 0

        cls_token = self.cls_token_mod0.repeat( x.shape[0], x.shape[1], 1, 1, 1)
        x = torch.cat([cls_token, x], dim=2)
        x = self.inner_tf_mod0_l3(x)
        x = x[:, :, 0].unsqueeze(dim=2)

        x = self.outer_positional_embedding(x)

        x = self.outer_tf_mod0_l3(x)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_concat_l1_h2_RA_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128*2#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=2)

        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=2)


        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(128, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(128, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        x = torch.cat([xeeg, xeog], dim=-1)


        # Inner layer 0

        cls_token = self.cls_token_mod0.repeat( x.shape[0], x.shape[1], 1, 1, 1)
        x = torch.cat([cls_token, x], dim=2)
        x = self.inner_tf_mod0_l3(x)
        x = x[:, :, 0].unsqueeze(dim=2)

        x = self.outer_positional_embedding(x)

        x = self.outer_tf_mod0_l3(x)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_concat_EEG_EOG_EMG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128*3#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat
        xemg = x[2][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")
        xemg = einops.rearrange(xemg,"b outer mod f inner -> b outer inner mod f")

        x = torch.cat([xeeg, xeog, xemg], dim=-1)

        x = self.tf(x)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_dex_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 5, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

        self.avg =  nn.AvgPool1d(kernel_size=5)
        self.avg_outer =  nn.AvgPool1d(kernel_size=7)

    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        b, outer, mod, f, inner = xeeg.shape

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        # Inner layer 0
        xeeg = self.inner_tf_mod0_l0(xeeg)
        # common = self.avg(xeeg)
        common = einops.rearrange(self.avg( einops.rearrange(xeeg,"b outer inner mod f -> (b outer mod) f inner")), " (b outer mod) f inner -> b outer inner mod f", b=b, outer=outer, mod=mod)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        xeog, common = xeog[:,:,:-5], einops.rearrange(self.avg( einops.rearrange(xeog,"b outer inner mod f -> (b outer mod) f inner")), " (b outer mod) f inner -> b outer inner mod f", b=b, outer=outer, mod=mod)

        # Inner layer 1
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:, :, :-5], einops.rearrange(self.avg( einops.rearrange(xeeg,"b outer inner mod f -> (b outer mod) f inner")), " (b outer mod) f inner -> b outer inner mod f", b=b, outer=outer, mod=mod)

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog, common = xeog[:, :, :-5], einops.rearrange(self.avg( einops.rearrange(xeog,"b outer inner mod f -> (b outer mod) f inner")), " (b outer mod) f inner -> b outer inner mod f", b=b, outer=outer, mod=mod)

        # Inner layer 2
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, :-5], einops.rearrange(self.avg( einops.rearrange(xeeg,"b outer inner mod f -> (b outer mod) f inner")), " (b outer mod) f inner -> b outer inner mod f", b=b, outer=outer, mod=mod)

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, :-5], einops.rearrange(self.avg( einops.rearrange(xeog,"b outer inner mod f -> (b outer mod) f inner")), " (b outer mod) f inner -> b outer inner mod f", b=b, outer=outer, mod=mod)

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), einops.rearrange(self.avg( einops.rearrange(xeeg,"b outer inner mod f -> (b outer mod) f inner")), " (b outer mod) f inner -> b outer inner mod f", b=b, outer=outer, mod=mod)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)


        # Outer layer 0

        xeeg = self.outer_tf_mod0_l0(xeeg)
        common = einops.rearrange(self.avg_outer(einops.rearrange(xeeg, "b outer inner mod f -> (b inner mod) f outer")), " (b inner mod) f outer -> b outer inner mod f", b=b, inner=1, mod=mod)

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l0(xeog)
        xeog, common = xeog[:, :-3], einops.rearrange(self.avg_outer(einops.rearrange(xeog[:, :-3], "b outer inner mod f -> (b inner mod) f outer")), " (b inner mod) f outer -> b outer inner mod f", b=b, inner=1, mod=mod)

        # Outer layer 1

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:, :-3], einops.rearrange(self.avg_outer(einops.rearrange(xeeg[:, :-3], "b outer inner mod f -> (b inner mod) f outer")), " (b inner mod) f outer -> b outer inner mod f", b=b, inner=1, mod=mod)

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l1(xeog)
        xeog, common = xeog[:, :-3], einops.rearrange(self.avg_outer(einops.rearrange(xeog[:, :-3], "b outer inner mod f -> (b inner mod) f outer")), " (b inner mod) f outer -> b outer inner mod f", b=b, inner=1, mod=mod)

        # Outer layer 2

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :-3], einops.rearrange(self.avg_outer(einops.rearrange(xeeg[:, :-3], "b outer inner mod f -> (b inner mod) f outer")), " (b inner mod) f outer -> b outer inner mod f", b=b, inner=1, mod=mod)

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :-3], einops.rearrange(self.avg_outer(einops.rearrange(xeog[:, :-3], "b outer inner mod f -> (b inner mod) f outer")), " (b inner mod) f outer -> b outer inner mod f", b=b, inner=1, mod=mod)

        # Outer layer 3

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :-3], einops.rearrange(self.avg_outer(einops.rearrange(xeeg[:, :-3], "b outer inner mod f -> (b inner mod) f outer")), " (b inner mod) f outer -> b outer inner mod f", b=b, inner=1, mod=mod)

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog = xeog[:, :-3]

        x = torch.cat([xeeg, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_real_dex_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 5, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

        self.avg =  nn.AvgPool1d(kernel_size=5)
        self.avg_outer =  nn.AvgPool1d(kernel_size=7)

    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        b, outer, mod, f, inner = xeeg.shape

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        # Inner layer 0
        xeeg = self.inner_tf_mod0_l0(xeeg)
        # common = self.avg(xeeg)
        common_eeg_ex = einops.rearrange(self.avg( einops.rearrange(xeeg,"b outer inner mod f -> (b outer mod) f inner")), " (b outer mod) f inner -> b outer inner mod f", b=b, outer=outer, mod=mod)

        xeog = self.inner_tf_mod1_l0(xeog)
        common_eog = einops.rearrange(self.avg( einops.rearrange(xeog,"b outer inner mod f -> (b outer mod) f inner")), " (b outer mod) f inner -> b outer inner mod f", b=b, outer=outer, mod=mod)

        # Inner layer 1
        xeeg = torch.cat([xeeg, common_eog], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg, common_eeg = xeeg[:, :, :-5], einops.rearrange(self.avg( einops.rearrange(xeeg,"b outer inner mod f -> (b outer mod) f inner")), " (b outer mod) f inner -> b outer inner mod f", b=b, outer=outer, mod=mod)

        xeog = torch.cat([xeog, common_eeg_ex],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog, common_eog = xeog[:, :, :-5], einops.rearrange(self.avg( einops.rearrange(xeog,"b outer inner mod f -> (b outer mod) f inner")), " (b outer mod) f inner -> b outer inner mod f", b=b, outer=outer, mod=mod)

        common_eeg_ex = common_eeg
        # Inner layer 2
        xeeg = torch.cat([xeeg, common_eog], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common_eeg = xeeg[:, :, :-5], einops.rearrange(self.avg( einops.rearrange(xeeg,"b outer inner mod f -> (b outer mod) f inner")), " (b outer mod) f inner -> b outer inner mod f", b=b, outer=outer, mod=mod)

        xeog = torch.cat([xeog, common_eeg_ex],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common_eog = xeog[:, :, :-5], einops.rearrange(self.avg( einops.rearrange(xeog,"b outer inner mod f -> (b outer mod) f inner")), " (b outer mod) f inner -> b outer inner mod f", b=b, outer=outer, mod=mod)
        common_eeg_ex = common_eeg

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common_eog], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog, common_eeg_ex],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)


        # Outer layer 0

        xeeg = self.outer_tf_mod0_l0(xeeg)
        common_eeg_ex = einops.rearrange(self.avg_outer(einops.rearrange(xeeg, "b outer inner mod f -> (b inner mod) f outer")), " (b inner mod) f outer -> b outer inner mod f", b=b, inner=1, mod=mod)

        xeog = self.outer_tf_mod1_l0(xeog)
        common_eog = einops.rearrange(self.avg_outer(einops.rearrange(xeog, "b outer inner mod f -> (b inner mod) f outer")), " (b inner mod) f outer -> b outer inner mod f", b=b, inner=1, mod=mod)

        # Outer layer 1

        xeeg = torch.cat([xeeg, common_eog], dim=1)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg, common_eeg = xeeg[:, :-3], einops.rearrange(self.avg_outer(einops.rearrange(xeeg[:, :-3], "b outer inner mod f -> (b inner mod) f outer")), " (b inner mod) f outer -> b outer inner mod f", b=b, inner=1, mod=mod)

        xeog = torch.cat([xeog, common_eeg_ex], dim=1)
        xeog = self.outer_tf_mod1_l1(xeog)
        xeog, common_eog = xeog[:, :-3], einops.rearrange(self.avg_outer(einops.rearrange(xeog[:, :-3], "b outer inner mod f -> (b inner mod) f outer")), " (b inner mod) f outer -> b outer inner mod f", b=b, inner=1, mod=mod)

        common_eeg_ex = common_eeg
        # Outer layer 2

        xeeg = torch.cat([xeeg, common_eog], dim=1)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common_eeg = xeeg[:, :-3], einops.rearrange(self.avg_outer(einops.rearrange(xeeg[:, :-3], "b outer inner mod f -> (b inner mod) f outer")), " (b inner mod) f outer -> b outer inner mod f", b=b, inner=1, mod=mod)

        xeog = torch.cat([xeog, common_eeg_ex], dim=1)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common_eog = xeog[:, :-3], einops.rearrange(self.avg_outer(einops.rearrange(xeog[:, :-3], "b outer inner mod f -> (b inner mod) f outer")), " (b inner mod) f outer -> b outer inner mod f", b=b, inner=1, mod=mod)

        common_eeg_ex = common_eeg

        # Outer layer 3

        xeeg = torch.cat([xeeg, common_eog], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :-3]

        xeog = torch.cat([xeog, common_eeg_ex], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog = xeog[:, :-3]

        x = torch.cat([xeeg, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_dex_pcommon_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.fc_common_1 = nn.Sequential(
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, d_model),
                        nn.ReLU()
                    )
        self.fc_common_2 = nn.Sequential(
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, d_model),
                        nn.ReLU()
                    )
        self.fc_common_3 = nn.Sequential(
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, d_model),
                        nn.ReLU()
                    )
        self.fc_common_4 = nn.Sequential(
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, d_model),
                        nn.ReLU()
                    )
        self.fc_common_outer_1 = nn.Sequential(
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, d_model),
                        nn.ReLU()
                    )
        self.fc_common_outer_2 = nn.Sequential(
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, d_model),
                        nn.ReLU()
                    )
        self.fc_common_outer_3 = nn.Sequential(
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, d_model),
                        nn.ReLU()
                    )

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 5, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

        # self.avg =  nn.AvgPool1d(kernel_size=)
        self.avg_outer =  nn.AvgPool1d(kernel_size=7)

    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        b, outer, mod, f, inner = xeeg.shape

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        # Inner layer 0
        xeeg = self.inner_tf_mod0_l0(xeeg)
        common_eeg = einops.rearrange(xeeg,"b outer inner mod f -> (b outer mod) f inner").mean(dim=-1)

        xeog = self.inner_tf_mod1_l0(xeog)
        common_eog = einops.rearrange(xeog,"b outer inner mod f -> (b outer mod) f inner").mean(dim=-1)

        common = einops.rearrange(self.fc_common_1(torch.cat([common_eeg, common_eog], dim=1)), "(b outer inner mod) f -> b outer inner mod f",  b=b, outer=outer, mod=1, inner=1)

        # Inner layer 1
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg, common_eeg = xeeg[:, :, :-1], einops.rearrange(xeeg,"b outer inner mod f -> (b outer mod) f inner").mean(dim=-1)

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog, common_eog = xeog[:, :, :-1], einops.rearrange(xeog,"b outer inner mod f -> (b outer mod) f inner").mean(dim=-1)

        common = einops.rearrange(self.fc_common_2(torch.cat([common_eeg, common_eog], dim=1)), "(b outer inner mod) f -> b outer inner mod f",  b=b, outer=outer, mod=1, inner=1)


        # Inner layer 2
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common_eeg = xeeg[:, :, :-1], einops.rearrange(xeeg,"b outer inner mod f -> (b outer mod) f inner").mean(dim=-1)

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common_eog = xeog[:, :, :-1], einops.rearrange(xeog,"b outer inner mod f -> (b outer mod) f inner").mean(dim=-1)

        common = einops.rearrange(self.fc_common_3(torch.cat([common_eeg, common_eog], dim=1)), "(b outer inner mod) f -> b outer inner mod f",  b=b, outer=outer, mod=1, inner=1)

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common_eeg = xeeg[:, :, 0].unsqueeze(dim=2), einops.rearrange(xeeg,"b outer inner mod f -> (b outer mod) f inner").mean(dim=-1)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common_eog = xeeg[:, :, 0].unsqueeze(dim=2),  einops.rearrange(xeog,"b outer inner mod f -> (b outer mod) f inner").mean(dim=-1)

        common = einops.rearrange(self.fc_common_4(torch.cat([common_eeg, common_eog], dim=1)), "(b outer inner mod) f -> b outer inner mod f",  b=b, outer=outer, mod=1, inner=1).mean(dim=1).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)

        # Outer layer 0

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        xeeg, common_eeg = xeeg[:, :-1], einops.rearrange(xeeg,"b outer inner mod f -> (b inner mod) f outer").mean(dim=-1)

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l0(xeog)
        xeog, common_eog = xeog[:, :-1], einops.rearrange(xeog,"b outer inner mod f -> (b inner mod) f outer").mean(dim=-1)

        common = einops.rearrange(self.fc_common_outer_1(torch.cat([common_eeg, common_eog], dim=1)), "(b outer inner mod) f -> b outer inner mod f",  b=b, outer=1, mod=1, inner=1)
        # Outer layer 1

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg, common_eeg = xeeg[:, :-1], einops.rearrange(xeeg,"b outer inner mod f -> (b inner mod) f outer").mean(dim=-1)

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l1(xeog)
        xeog, common_eog = xeog[:, :-1], einops.rearrange(xeog,"b outer inner mod f -> (b inner mod) f outer").mean(dim=-1)

        common = einops.rearrange(self.fc_common_outer_2(torch.cat([common_eeg, common_eog], dim=1)), "(b outer inner mod) f -> b outer inner mod f",  b=b, outer=1, mod=1, inner=1)
        # Outer layer 2

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common_eeg = xeeg[:, :-1], einops.rearrange(xeeg,"b outer inner mod f -> (b inner mod) f outer").mean(dim=-1)

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common_eog = xeog[:, :-1], einops.rearrange(xeog,"b outer inner mod f -> (b inner mod) f outer").mean(dim=-1)

        common = einops.rearrange(self.fc_common_outer_3(torch.cat([common_eeg, common_eog], dim=1)), "(b outer inner mod) f -> b outer inner mod f",  b=b, outer=1, mod=1, inner=1)

        # Outer layer 3

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :-1]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog = xeog[:, :-1]

        x = torch.cat([xeeg, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_suppmod_lim2_c5_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 5, 1, d_model))
        self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 5, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        # Inner layer 0
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        # xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        # xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        # Inner layer 3

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -5:]

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -5:]

        # common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)

        common =  self.common_bottleneck_outer.repeat( xeeg.shape[0], 1, 1, 1, 1)

        # Outer layer 0

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l0(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 1

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 2

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :-5], xeog[:, -5:]

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :-5], xeeg[:, -5:]

        # Outer layer 3

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :-5], xeog[:, -5:]

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :-5], xeeg[:, -5:].mean(dim=1).unsqueeze(dim=1)

        common =  common.repeat( 1, xeeg.shape[1], 1, 1, 1)

        x = torch.cat([xeeg, common], dim=2)
        # x = xeeg

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_suppmod_lim2_c1_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 5, 1, d_model))
        self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 5, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        # Inner layer 0
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        # xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        # xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, :-1], xeog[:, :, -1:]

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, :-1], xeeg[:, :, -1:]

        # Inner layer 3

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, 1, 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -1:]

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -1:]

        # common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)


        # Outer layer 0

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l0(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 1

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 2

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :-1], xeog[:, -1:]

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :-1], xeeg[:, -1:]

        # Outer layer 3

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :-1], xeog[:, -1:]

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :-1], xeeg[:, -1:]

        common =  common.repeat( 1, xeeg.shape[1], 1, 1, 1)

        x = torch.cat([xeeg, common], dim=2)
        # x = xeeg

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_suppcontext_lim2_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        # Inner layer 0
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        # xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        # xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, :-1], xeog[:, :, -1:]

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, :-1], xeeg[:, :, -1:]


        # Inner layer 3

        xeog = torch.cat([ xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, :-1], xeog[:, :, -1:]

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, :-1], xeeg[:, :, -1:]


        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, 1, 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -1:]

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -1:]

        # common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)

        # common =  self.common_bottleneck_outer.repeat( xeeg.shape[0], 1, 1, 1, 1)

        # Outer layer 0

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l0(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 1

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 2

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common = xeog[:, : common.shape[1]], xeog[:, common.shape[1]:]

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, : common.shape[1]], xeeg[:,  common.shape[1]:]

        # Outer layer 3

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, : common.shape[1]], xeog[:,  common.shape[1]:]

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, : common.shape[1]], xeeg[:,  common.shape[1]:]

        # x = torch.cat([xeeg, common], dim=2)
        x = common

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x


class EEG_SLEEP_MultiModal_bottleneck_c5_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 5, 1, d_model))
        self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 5, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        # Inner layer 0
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -5:]

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, 1, 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -5:]


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)


        # Outer layer 0

        common =  self.common_bottleneck_outer.repeat( xeeg.shape[0], 1, 1, 1, 1)

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        xeeg, common = xeeg[:, :-5], xeeg[:, -5:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l0(xeog)
        xeog, common = xeog[:, :-5], xeog[:, -5:]

        # Outer layer 1

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:, :-5], xeeg[:, -5:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l1(xeog)
        xeog, common = xeog[:, :-5], xeog[:, -5:]

        # Outer layer 2

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :-5], xeeg[:, -5:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :-5], xeog[:, -5:]

        # Outer layer 3

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :-5], xeeg[:, -5:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :-5], xeog[:, -5:]
        #
        x = torch.cat([xeeg, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_bottleneck_lim2_c5_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 5, 1, d_model))
        self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 5, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        # Inner layer 0
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        # xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        # xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -5:]

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -5:]


        # common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)


        # Outer layer 0

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l0(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 1

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 2

        common =  self.common_bottleneck_outer.repeat( xeeg.shape[0], 1, 1, 1, 1)

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :-5], xeeg[:, -5:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :-5], xeog[:, -5:]

        # Outer layer 3

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :-5], xeeg[:, -5:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :-5], xeog[:, -5:]
        #
        x = torch.cat([xeeg, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_bottleneck_lim2_c1_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        # Inner layer 0
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        # xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        # xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2
        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, :-1], xeeg[:, :, -1:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, :-1], xeog[:, :, -1:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -1:]

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -1:]


        # common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)


        # Outer layer 0

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l0(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 1

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 2
        common =  self.common_bottleneck_outer.repeat( xeeg.shape[0],1, 1, 1, 1)

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :-1], xeeg[:, -1:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :-1], xeog[:, -1:]

        # Outer layer 3

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :-1], xeeg[:, -1:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :-1], xeog[:, -1:]
        #
        x = torch.cat([xeeg, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_bottleneck_l1_c1_RA_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        # Inner layer 0

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -1:]

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, 1, 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -1:]


        # common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)


        # Outer layer 2

        common =  self.common_bottleneck_outer.repeat( xeeg.shape[0],1, 1, 1, 1)

        # Outer layer 3

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :-1], xeeg[:,-1:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :-1], xeog[:,-1:]
        #

        x = torch.cat([xeeg, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_bottleneck_l1_c5_RA_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, heads=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 5, 1, d_model))
        self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 5, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        # Inner layer 0

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -5:]

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, 1, 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -5:]


        # common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)


        # Outer layer 2

        common =  self.common_bottleneck_outer.repeat( xeeg.shape[0], 1, 1, 1, 1)

        # Outer layer 3

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :-5], xeeg[:,-5:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :-5], xeog[:,-5:]
        #

        x = torch.cat([xeeg, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_ibottleneck_lim2_c1_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        # Inner layer 0
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        # xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        # xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, :-1], xeeg[:, :, -1:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, :-1], xeog[:, :, -1:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -1:]

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -1:]


        # common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)

        # Outer layer 0

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l0(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 1

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 2

        xeeg = self.outer_tf_mod0_l2(xeeg)

        xeog = self.outer_tf_mod1_l2(xeog)

        # Outer layer 3

        xeeg = self.outer_tf_mod0_l3(xeeg)

        xeog = self.outer_tf_mod1_l3(xeog)
        #
        x = torch.cat([xeeg, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_obottleneck_lim2_c1_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        # common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        # Inner layer 0
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        # xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        # xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        # xeeg, common = xeeg[:, :, :-1], xeeg[:, :, -1:]

        # xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        # xeog, common = xeog[:, :, :-1], xeog[:, :, -1:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        # common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)


        # Outer layer 0

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l0(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 1

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 2

        common =  self.common_bottleneck_outer.repeat( xeeg.shape[0], 1, 1, 1, 1)

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :-1], xeeg[:, -1:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :-1], xeog[:, -1:]

        # Outer layer 3

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :-1], xeeg[:, -1:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :-1], xeog[:, -1:]
        #

        x = torch.cat([xeeg, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_bottleneck_lim1_c1_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        # Inner layer 0
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        # xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        # xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        # xeeg, common = xeeg[:, :, :-1], xeeg[:, :, -1:]

        # xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        # xeog, common = xeog[:, :, :-1], xeog[:, :, -1:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -1:]

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -1:]


        # common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)


        # Outer layer 0

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l0(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 1

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 2

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l2(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 3

        common =  self.common_bottleneck_outer.repeat( xeeg.shape[0], 1, 1, 1, 1)

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :-1], xeeg[:, -1:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :-1], xeog[:, -1:]
        #
        x = torch.cat([xeeg, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_bottleneck_lim0_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]
        encoder_filters = 2
        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        # common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        # Inner layer 0
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        # xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        # xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        # xeeg, common = xeeg[:, :, :-1], xeeg[:, :, -1:]

        # xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        # xeog, common = xeog[:, :, :-1], xeog[:, :, -1:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        # common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)

        # common =  self.common_bottleneck_outer.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)

        # Outer layer 0

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l0(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 1

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 2

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l2(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 3

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l3(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)
        #
        x = torch.cat([xeeg, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x
class SleepEnc_bottleneck_lim0_EEG_EOG(nn.Module):
    def __init__(self, args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args
        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if "dim_proj" in args:
            dim_proj = args.dim_proj
        else:
            dim_proj = d_model
        if "num_layers" in args:
            num_layers = args.num_layers
        else:
            num_layers = 1

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_mod0_l0 = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=num_layers)
        self.inner_tf_mod0_l1 = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=num_layers)
        self.inner_tf_mod0_l2 = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=num_layers)
        self.inner_tf_mod0_l3 = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=num_layers)

        self.inner_tf_mod1_l0 = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=num_layers)
        self.inner_tf_mod1_l1 = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=num_layers)
        self.inner_tf_mod1_l2 = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=num_layers)
        self.inner_tf_mod1_l3 = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=num_layers)

        self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, dim_proj=dim_proj, num_layers=num_layers)
        self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, dim_proj=dim_proj, num_layers=num_layers)
        self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, dim_proj=dim_proj, num_layers=num_layers)
        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, dim_proj=dim_proj, num_layers=num_layers)

        self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, dim_proj=dim_proj, num_layers=num_layers)
        self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, dim_proj=dim_proj, num_layers=num_layers)
        self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, dim_proj=dim_proj, num_layers=num_layers)
        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, dim_proj=dim_proj, num_layers=num_layers)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog,"b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        # common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        # Inner layer 0
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        # xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        # xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        # xeeg, common = xeeg[:, :, :-1], xeeg[:, :, -1:]

        # xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        # xeog, common = xeog[:, :, :-1], xeog[:, :, -1:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        # common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)

        # common =  self.common_bottleneck_outer.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)

        # Outer layer 0

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l0(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 1

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 2

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l2(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 3

        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l3(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)
        #
        x = torch.cat([xeeg, xeog], dim=2)

        return x

class EEG_SLEEP_MultiModal_bottleneck_EEG_EOG_EMG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*4, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod2_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod2_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod2_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod2_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod2_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod2_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod2_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod2_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 5, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat
        xemg = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")
        xemg = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)
        xemg = self.inner_positional_embedding(xemg)

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        # Inner layer 0
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        xemg = torch.cat([xemg, common], dim=2)
        xemg = self.inner_tf_mod2_l0(xemg)
        xemg, common = xemg[:,:,:-5], xemg[:,:,-5:]

        # Inner layer 1
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        xemg = torch.cat([xemg, common], dim=2)
        xemg = self.inner_tf_mod2_l1(xemg)
        xemg, common = xemg[:,:,:-5], xemg[:,:,-5:]

        # Inner layer 2
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        xemg = torch.cat([xemg, common], dim=2)
        xemg = self.inner_tf_mod2_l2(xemg)
        xemg, common = xemg[:,:,:-5], xemg[:,:,-5:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -5:]

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -5:]

        cls_token_emg = self.cls_token_mod2.repeat( xemg.shape[0], xemg.shape[1], 1, xemg.shape[3], 1)
        xemg = torch.cat([cls_token_emg, xemg, common],dim=2)
        xemg = self.inner_tf_mod2_l3(xemg)
        xemg, common = xemg[:, :, 0].unsqueeze(dim=2), xemg[:, :, -5:]


        common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        xemg = self.outer_positional_embedding(xemg)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)


        # Outer layer 0

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l0(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        xemg = torch.cat([xemg, common], dim=2)
        xemg = self.outer_tf_mod2_l0(xemg)
        xemg, common = xemg[:, :, 0].unsqueeze(dim=2), xemg[:, :, 1].unsqueeze(dim=2)

        # Outer layer 1

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l1(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        xemg = torch.cat([xemg, common], dim=2)
        xemg = self.outer_tf_mod2_l1(xemg)
        xemg, common = xemg[:, :, 0].unsqueeze(dim=2), xemg[:, :, 1].unsqueeze(dim=2)

        # Outer layer 2

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        xemg = torch.cat([xemg, common], dim=2)
        xemg = self.outer_tf_mod2_l2(xemg)
        xemg, common = xemg[:, :, 0].unsqueeze(dim=2), xemg[:, :, 1].unsqueeze(dim=2)

        # Outer layer 3

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        xemg = torch.cat([xemg, common], dim=2)
        xemg = self.outer_tf_mod2_l3(xemg)
        xemg, common = xemg[:, :, 0].unsqueeze(dim=2), xemg[:, :, 1].unsqueeze(dim=2)
        #
        x = torch.cat([common, xeeg, xeog, xemg], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_sharedi_bottleneck_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*3, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        # self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 5, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        # Inner layer 0
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod0_l0(xeog)
        xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod0_l1(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod0_l2(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -5:]

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod0_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -5:]


        common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)


        # Outer layer 0

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l0(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 1

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l1(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 2

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 3

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)
        #
        x = torch.cat([xeeg, common, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_ibottleneck_merged_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*3, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        # self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 5, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        # Inner layer 0
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -5:]

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -5:]


        common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        x = torch.cat([xeeg, common, xeog], dim=2)
        x = self.outer_tf_mod0_l0(x)
        x = self.outer_tf_mod0_l1(x)
        x = self.outer_tf_mod0_l2(x)
        x = self.outer_tf_mod0_l3(x)


        # # Outer layer 0
        #
        # xeeg = torch.cat([xeeg, common], dim=2)
        # xeeg = self.outer_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)
        #
        # xeog = torch.cat([xeog, common], dim=2)
        # xeog = self.outer_tf_mod1_l0(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)
        #
        # # Outer layer 1
        #
        # xeeg = torch.cat([xeeg, common], dim=2)
        # xeeg = self.outer_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)
        #
        # xeog = torch.cat([xeog, common], dim=2)
        # xeog = self.outer_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)
        #
        # # Outer layer 2
        #
        # xeeg = torch.cat([xeeg, common], dim=2)
        # xeeg = self.outer_tf_mod0_l2(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)
        #
        # xeog = torch.cat([xeog, common], dim=2)
        # xeog = self.outer_tf_mod1_l2(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)
        #
        # # Outer layer 3
        #
        # xeeg = torch.cat([xeeg, common], dim=2)
        # xeeg = self.outer_tf_mod0_l3(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)
        #
        # xeog = torch.cat([xeog, common], dim=2)
        # xeog = self.outer_tf_mod1_l3(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)
        # #
        # x = torch.cat([xeeg, common, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiModal_οbottleneck_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        encoder_filters = 2
        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*3, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 5, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg = self.inner_tf_mod0_l2(xeeg)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg],dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        xeog = self.inner_tf_mod1_l0(xeog)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog = self.inner_tf_mod1_l2(xeog)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)


        # Outer layer 0

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l0(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 1

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l1(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 2

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 3

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)
        #
        x = torch.cat([xeeg, common, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x
class EEG_SLEEP_OBottleneck_EEG_EOG_Encoder(nn.Module):
    def __init__(self, args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        n_layers = 1

        self.inner_tf_mod0_l0 = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=n_layers)
        self.inner_tf_mod0_l1 = inner_ch_att_RA(d_model, pos=False, rpos=rpos,  inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=n_layers)
        self.inner_tf_mod0_l2 = inner_ch_att_RA(d_model, pos=False, rpos=rpos,  inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=n_layers)
        self.inner_tf_mod0_l3 = inner_ch_att_RA(d_model, pos=False, rpos=rpos,  inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=n_layers)

        self.inner_tf_mod1_l0 = inner_ch_att_RA(d_model, pos=False, rpos=rpos,  inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=n_layers)
        self.inner_tf_mod1_l1 = inner_ch_att_RA(d_model, pos=False, rpos=rpos,  inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=n_layers)
        self.inner_tf_mod1_l2 = inner_ch_att_RA(d_model, pos=False, rpos=rpos,  inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=n_layers)
        self.inner_tf_mod1_l3 = inner_ch_att_RA(d_model, pos=False, rpos=rpos,  inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=n_layers)

        self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model, pos=False, rpos=rpos,  inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=n_layers)
        self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model, pos=False, rpos=rpos,  inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=n_layers)
        self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model, pos=False, rpos=rpos,  inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=n_layers)
        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, rpos=rpos,  inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=n_layers)

        self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model, pos=False, rpos=rpos,  inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=n_layers)
        self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model, pos=False, rpos=rpos,  inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=n_layers)
        self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model, pos=False, rpos=rpos,  inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=n_layers)
        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, rpos=rpos,  inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=n_layers)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 5, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog,"b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg = self.inner_tf_mod0_l2(xeeg)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg],dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        xeog = self.inner_tf_mod1_l0(xeog)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog = self.inner_tf_mod1_l2(xeog)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)


        # Outer layer 0
        print(xeeg.shape)
        print(common.shape)
        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l0(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 1

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l1(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 2

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        # Outer layer 3

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)
        #
        x = torch.cat([xeeg, common, xeog], dim=2)

        return x

class EEG_SLEEP_MultiModal_ibottleneck_merged_EEG_EOG_EMG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*4, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod2_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod2_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod2_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod2_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod2_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod2_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod2_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod2_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 5, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat
        xemg = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")
        xemg = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)
        xemg = self.inner_positional_embedding(xemg)

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        # Inner layer 0
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        xemg = torch.cat([xemg, common], dim=2)
        xemg = self.inner_tf_mod2_l0(xemg)
        xemg, common = xemg[:,:,:-5], xemg[:,:,-5:]

        # Inner layer 1
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        xemg = torch.cat([xemg, common], dim=2)
        xemg = self.inner_tf_mod2_l1(xemg)
        xemg, common = xemg[:,:,:-5], xemg[:,:,-5:]

        # Inner layer 2
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        xemg = torch.cat([xemg, common], dim=2)
        xemg = self.inner_tf_mod2_l2(xemg)
        xemg, common = xemg[:,:,:-5], xemg[:,:,-5:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -5:]

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -5:]

        cls_token_emg = self.cls_token_mod2.repeat( xemg.shape[0], xemg.shape[1], 1, xemg.shape[3], 1)
        xemg = torch.cat([cls_token_emg, xemg, common],dim=2)
        xemg = self.inner_tf_mod2_l3(xemg)
        xemg, common = xemg[:, :, 0].unsqueeze(dim=2), xemg[:, :, -5:]


        common = common.mean(dim=2).unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)
        xemg = self.outer_positional_embedding(xemg)
        # common = self.outer_positional_embedding(common)


        # x = torch.cat([xeeg, common, xeog], dim=2)
        # x = self.outer_tf_mod0_l0(x)
        # x = self.outer_tf_mod0_l1(x)
        # x = self.outer_tf_mod0_l2(x)
        # x = self.outer_tf_mod0_l3(x)


        # Outer layer 0

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l0(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        xemg = torch.cat([xemg, common], dim=2)
        xemg = self.outer_tf_mod2_l0(xemg)
        xemg, common = xemg[:, :, 0].unsqueeze(dim=2), xemg[:, :, 1].unsqueeze(dim=2)

        # Outer layer 1

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l1(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        xemg = torch.cat([xemg, common], dim=2)
        xemg = self.outer_tf_mod2_l1(xemg)
        xemg, common = xemg[:, :, 0].unsqueeze(dim=2), xemg[:, :, 1].unsqueeze(dim=2)

        # Outer layer 2

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        xemg = torch.cat([xemg, common], dim=2)
        xemg = self.outer_tf_mod2_l2(xemg)
        xemg, common = xemg[:, :, 0].unsqueeze(dim=2), xemg[:, :, 1].unsqueeze(dim=2)

        # Outer layer 3

        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)

        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)

        xemg = torch.cat([xemg, common], dim=2)
        xemg = self.outer_tf_mod2_l3(xemg)
        xemg, common = xemg[:, :, 0].unsqueeze(dim=2), xemg[:, :, 1].unsqueeze(dim=2)
        #
        x = torch.cat([common, xeeg, xeog, xemg], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x


class EEG_SLEEP_GM_EEG(nn.Module):

    def __init__(self, encs=[None], args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  args.dmodel#64*8
        fc_inner = args.fc_inner
        num_classes = args.num_classes

        self.args = args
        self.num_encoders = 0
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
                self.num_encoders +=1


        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes)
                        # nn.Softmax(dim=1)
                    )
    def forward(self, x, inits= None, return_matches=False, extract_norm=False, return_inter_reps=False, return_final_reps=False, return_order=False):

        for i in range(self.num_encoders):
            enc = getattr(self, "enc_{}".format(i))
            x = enc(x, extract_norm=extract_norm)
        x = x[0]
        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        output = {"preds": {"combined": x}}

        return output

class EEG_SLEEP_GM_EEG_EOG(nn.Module):

    def __init__(self, encs=[None], args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
    def forward(self, x, inits= None, extract_norm=False):

        xeeg, xeog = self.enc_0(x, extract_norm=extract_norm)

        x = torch.cat([xeeg, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_GM_TC_EEG(nn.Module):

    def __init__(self, encs=[None], args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
    def forward(self, x, inits= None, extract_norm=False):

        _, x = self.enc_0(x, extract_norm=extract_norm)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_TimeShuffle_GM_EEG(nn.Module):

    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 2
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
    def forward(self, x, inits= None, extract_norm=False):

        x_cls, _ = self.enc_0(x, extract_norm=extract_norm)

        if len(x_cls.shape)>1:
            x_cls = x_cls.flatten(start_dim=1)
        x_cls = self.fc_out(x_cls)
        return x_cls
class EEG_TimeShuffle_Order_GM_EEG(nn.Module):

    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 21
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
    def forward(self, x, inits= None, extract_norm=False):

        x = self.enc_0(x, extract_norm=extract_norm)

        if len(x.shape)>1:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)

        x = self.fc_out(x)
        return x
class EEG_SLEEP_GM2_EEG(nn.Module):

    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
    def forward(self, x, inits= None, extract_norm=False):

        x = self.enc_0(x, extract_norm=extract_norm)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x
class EEG_SLEEP_GM_common_EEG_EOG(nn.Module):

    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
    def forward(self, x, inits= None, extract_norm=False):

        x = self.enc_0(x, extract_norm=extract_norm)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x
class EEG_SLEEP_GM1_EEG_EOG(nn.Module):

    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
    def forward(self, x, inits= None):

        x = self.enc_0(x)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x
class EEG_SLEEP_GM2_EEG_EOG(nn.Module):

    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
    def forward(self, x, inits= None, extract_norm=False):

        x = self.enc_0(x, extract_norm=extract_norm)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x
class EEG_SLEEP_GM4_EEG_EOG(nn.Module):

    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*4, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
    def forward(self, x, inits= None):

        xeeg, xeog = self.enc_0(x)

        x = torch.cat([xeeg, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_Channel_EEG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args
        d_model = 128  # 64*8

        # self.channel_tf_mod0 = channel_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, num_layers=4)

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, num_layers=4)

        biased = False

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased,
                                              num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_mod0_channel = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)

        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :1, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding_0(xeeg)

        cls_token_eeg = self.cls_token_mod0.repeat(xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], xeeg.shape[4], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        # cls_token_eeg_channel = self.cls_token_mod0.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, 1, 1)
        # xeeg = torch.cat([cls_token_eeg_channel, xeeg], dim=4)
        # xeeg = self.channel_tf_mod0(xeeg, extract_norm=extract_norm)
        # xeeg = xeeg[:, :, :, :, 0].unsqueeze(dim=4)

        xeeg = self.outer_positional_embedding(xeeg)

        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        return xeeg
class EEG_SLEEP_TimeShuffle_EEG(nn.Module):
    def __init__(self, args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        self.args = args

        d_model = 128  # 64*8

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, num_layers=4)

        biased = False

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased,
                                              num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)

        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding_0(xeeg)

        cls_token_eeg = self.cls_token_mod0.repeat(xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1).unsqueeze(dim=2)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)

        cls_token_eeg_outer = self.cls_token_mod0_outer.repeat(xeeg.shape[0], 1, 1, xeeg.shape[3], 1).unsqueeze(dim=2)
        xeeg = torch.cat([cls_token_eeg_outer, xeeg], dim=1)
        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg_cls, xeeg  = xeeg[:, 0].unsqueeze(dim=1), xeeg[:, 1:]

        return xeeg_cls, xeeg
class EEG_SLEEP_TimeShuffle_Order_EEG(nn.Module):
    def __init__(self, args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        self.args = args

        d_model = 128  # 64*8

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, num_layers=4)

        biased = False

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased,
                                              num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)

        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding_0(xeeg)

        cls_token_eeg = self.cls_token_mod0.repeat(xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1).unsqueeze(dim=2)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)

        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        return xeeg

class EEG_SLEEP_Contrastive_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = 128  # 64*8

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, num_layers=4)

        self.inner_tf_mod1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, num_layers=4)

        biased = False

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased,
                                              num_layers=4)

        self.outer_tf_mod1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased,
                                              num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog, "b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        cls_token_eeg = self.cls_token_mod0.repeat(xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat(xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)
        xeog = self.inner_tf_mod1(xeog, extract_norm=extract_norm)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)
        xeog = self.outer_tf_mod1(xeog, extract_norm=extract_norm)

        return xeeg, xeog
class SleepEnc_Late_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)
        self.inner_tf_mod1 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)
        self.outer_tf_mod1 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_mod0.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)
        xeog = self.inner_tf_mod1(xeog, extract_norm=extract_norm)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)
        xeog = self.outer_tf_mod0(xeog, extract_norm=extract_norm)

        return torch.cat([xeeg, xeog], dim=-1)
class SleepEnc_Merged_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        rpos = args.rpos if "rpos" in args else False
        self.pos = args.pos if "pos" in args else True

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1, npoints=400)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1, npoints=400)

    def forward(self, x, inits=None, extract_norm=False, return_matches=False, return_inter_reps=False, return_order=False):

        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.pos:

            xeeg = self.inner_positional_embedding(xeeg)
            xeog = self.inner_positional_embedding(xeog)

        cls_token = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token, xeeg, xeog], dim=2)
        xeeg = self.inner_tf(xeeg, extract_norm=extract_norm)
        x = xeeg[:, :, 0].unsqueeze(dim=2)

        if self.pos:
            x = self.outer_positional_embedding(x)

        x = self.outer_tf(x, extract_norm=extract_norm)
        output = [x]
        if return_inter_reps:
            output.append(xeeg[:,:,1:])
        return output
class SleepEnc_LocGlob_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf = inner_locglob_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        x = torch.cat([xeeg, xeog], dim=-2)
        cls_token = self.cls_token.repeat(x.shape[0], x.shape[1], 1, 1, x.shape[4], 1)
        x = torch.cat([cls_token, x], dim=2)
        x = self.inner_tf(x, extract_norm=extract_norm)
        x = x[:, :, :1]

        x = self.outer_positional_embedding(x)

        x = self.outer_tf(x, extract_norm=extract_norm)

        return x
class SleepEnc_Merged_HPFC_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf = inner_ch_att_HPFC_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=2, gbiased=inner_biased, num_layers=4)

        self.outer_tf = outer_mod_att_HPFC_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=2, gbiased=outer_biased, num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_mod0.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        cls_token_eog = self.cls_token_mod1.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)

        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeog = torch.cat([cls_token_eeg, xeog], dim=2)

        [xeeg, xeog] = self.inner_tf([xeeg, xeog], extract_norm=extract_norm)
        xeeg, xeog = xeeg[:,:,:1], xeog[:,:,:1]

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        [xeeg, xeog] = self.outer_tf([xeeg, xeog], extract_norm=extract_norm)

        x = torch.cat([xeeg, xeog],dim=-1)

        return x
class SleepEnc_Bottleneck_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)

        self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)

        self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=1)
        self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=1)
        self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=1)
        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=1)

        self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=1)
        self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=1)
        self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=1)
        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=1)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.common_outer = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.common_inner = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        common =  self.common_inner.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1, 1)
        # Inner layer 0
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg, common = xeeg[:,:,:-1], xeeg[:,:,-1:]


        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        xeog, common = xeog[:,:,:-1], xeog[:,:,-1:]

        # Inner layer 1
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:,:,:-1], xeeg[:,:,-1:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog, common = xeog[:,:,:-1], xeog[:,:,-1:]

        # Inner layer 2
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:,:,:-1], xeeg[:,:,-1:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common = xeog[:,:,:-1], xeog[:,:,-1:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1, 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:,:,:1], xeeg[:,:,-1:]

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, 1, 1, 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:,:,:1], xeog[:,:,-1:]

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        common =  self.common_outer.repeat( xeeg.shape[0], 1, 1, 1, 1, 1)

        # Outer layer 0

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l0(xeeg)
        xeeg, common = xeeg[:, :-1],  xeeg[:, -1:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l0(xeog)
        xeog, common = xeog[:, :-1],  xeog[:, -1:]

        # Outer layer 1

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:, :-1],  xeeg[:, -1:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l1(xeog)
        xeog, common = xeog[:, :-1],  xeog[:, -1:]

        # Outer layer 2

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :-1],  xeeg[:, -1:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :-1],  xeog[:, -1:]

        # Outer layer 3

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :-1],  xeeg[:, -1:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :-1],  xeog[:, -1:]

        x = torch.cat([xeeg, xeog], dim=2)

        return x
class SleepEnc_Merged_Channels_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        x = torch.cat([xeeg, xeog], dim=4).flatten(start_dim=2, end_dim=4).unsqueeze(dim=3).unsqueeze(dim=3)

        cls_token_eeg = self.cls_token.repeat(x.shape[0], x.shape[1], 1, 1, x.shape[3], 1)
        x = torch.cat([cls_token_eeg, x], dim=2)
        x = self.inner_tf(x, extract_norm=extract_norm)
        x = x[:, :, 0].unsqueeze(dim=2)

        x = self.outer_positional_embedding(x)

        x = self.outer_tf(x, extract_norm=extract_norm)

        return x
class SleepEnc_EEG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos


        self.pos = args.pos if "pos" in args else True
        dropout = args.dropout if "dropout" in args else 0.1

        if "dim_proj" in args:
            dim_proj = args.dim_proj
        else:
            dim_proj = d_model
        if "num_layers" in args:
            num_layers = args.num_layers
        else:
            num_layers = 4

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dropout=dropout, gbiased=inner_biased, dim_proj=dim_proj, num_layers=num_layers)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dropout=dropout, gbiased=outer_biased, dim_proj=dim_proj, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        if self.pos == "trained":
            self.pos_eeg = pos_embedding(max_pos=200, dim=d_model)
        elif self.pos == "sinusoidal":
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                              channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                            channels=1)

    def forward(self, xeeg, extract_norm=False, **kwargs):
        xeeg = xeeg["stft_eeg"][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.pos == "trained":
            xeeg = self.pos_eeg.forward_inner(xeeg)
        elif self.pos =="sinusoidal":
            xeeg = self.inner_positional_embedding(xeeg)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf(xeeg, extract_norm=extract_norm)
        x = xeeg[:, :, 0].unsqueeze(dim=2)

        if self.pos == "trained":
            x = self.pos_eeg.forward_outer(x)
        elif self.pos =="sinusoidal":
            x = self.outer_positional_embedding(x)

        x = self.outer_tf(x, extract_norm=extract_norm)

        return {"output_features":x}
class SleepEnc_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos


        self.pos = args.pos if "pos" in args else True

        if "dim_proj" in args:
            dim_proj = args.dim_proj
        else:
            dim_proj = d_model
        if "num_layers" in args:
            num_layers = args.num_layers
        else:
            num_layers = 4

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, dim_proj=dim_proj, num_layers=num_layers)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, dim_proj=dim_proj, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        if self.pos:
            self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                              channels=1)
            self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                            channels=1)

    def forward(self, xeog, extract_norm=False, **kwargs):
        xeog = xeog["stft_eog"][:, :, :, :, 1:, :]  # mat

        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        if self.pos:
            xeog = self.inner_positional_embedding(xeog)

        cls_token_eog = self.cls_token.repeat(xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)
        xeog = self.inner_tf(xeog, extract_norm=extract_norm)
        x = xeog[:, :, 0].unsqueeze(dim=2)

        if self.pos:
            x = self.outer_positional_embedding(x)

        x = self.outer_tf(x, extract_norm=extract_norm)

        return {"output_features":x}

class pos_embedding(nn.Module):
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

class SleepEnc_ΕΕG_Time_OnlyCNN(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        ln_first = args.ln_first if "ln_first" in args else False
        rpos = args.rpos if "rpos" in args else False
        dim_proj = args.dim_proj if "dim_proj" in args else 128

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        # self.conv = nn.Sequential(
        #         nn.Conv1d(1, 32, kernel_size=10, padding=1),
        #         # nn.GroupNorm(num_groups=128, num_channels=128, affine=True),
        #         nn.GELU(),
        #         nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=1),
        #         nn.GELU(),
        #         nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
        #         nn.GELU(),
        #         nn.MaxPool1d(kernel_size=2),
        #         nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
        #         nn.GELU(),
        #         nn.MaxPool1d(kernel_size=2),
        #         nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
        #         nn.GELU(),
        #         nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
        #         nn.GELU()
        # )
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size= 5, stride=2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=5),
            nn.ReLU()
        )

    def forward(self, x, **kwargs):

        xeeg = x["time_eeg"] # mat

        b, outer = xeeg.shape[0], xeeg.shape[1]
        xeeg = einops.rearrange(xeeg, "b outer time ch -> (b outer) ch time")
        xeeg = self.conv(xeeg)
        xeeg = einops.rearrange(xeeg, " (b outer mod ch) k time -> b outer time mod ch k", b=b, outer=outer,  mod=1, ch=1)
        # print(xeeg.shape)
        xeeg = xeeg.mean(dim=2)
        output = {"output_features":{"combined":xeeg}}
        return output

class SleepEnc_ΕΕG_Time(nn.Module):
    def __init__(self, args, encs=[None]):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased = args.inner_biased if "inner_biased" in args else False
        outer_biased = args.outer_biased if "outer_biased" in args else False
        ln_first = args.ln_first if "ln_first" in args else False
        rpos = args.rpos if "rpos" in args else False
        dim_proj = args.dim_proj if "dim_proj" in args else 128

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=10, padding=1),
                # nn.GroupNorm(num_groups=128, num_channels=128, affine=True),
                nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=1),
                nn.GELU(),
                nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.GELU()
        )

        self.inner_tf_eeg = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=inner_biased, num_layers=4, ln_first=ln_first)
        self.outer_tf_eeg = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, dim_proj=dim_proj, gbiased=outer_biased, num_layers=4, ln_first=ln_first)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, **kwargs):

        xeeg = x[0] # mat

        b, outer = xeeg.shape[0], xeeg.shape[1]
        xeeg = einops.rearrange(xeeg, "b outer time ch -> (b outer) ch time")
        xeeg = self.conv(xeeg)
        xeeg = einops.rearrange(xeeg, " (b outer mod ch) k time -> b outer time mod ch k", b=b, outer=outer,  mod=1, ch=1)


        # xeeg = einops.rearrange(xeeg, "b outer time -> b outer time mod ch f", mod=1,ch=1)

        xeeg = self.inner_positional_embedding(xeeg)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        xeeg = self.inner_tf_eeg(xeeg)
        xeeg = xeeg[:, :, :1]

        xeeg = self.outer_positional_embedding(xeeg)
        xeeg = self.outer_tf_eeg(xeeg)

        output = [xeeg]

        return output
class SleepEnc_ΕΕG_Decoder(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if "dim_proj" in args:
            dim_proj = args.dim_proj
        else:
            dim_proj = d_model
        if "num_layers" in args:
            num_layers = args.num_layers
        else:
            num_layers = 4

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, dim_proj=dim_proj, num_layers=num_layers)

        self.outer_tf = outer_decoder_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, dim_proj=dim_proj, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.output_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model), requires_grad=True)

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf(xeeg, extract_norm=extract_norm)
        x = xeeg[:, :, 0].unsqueeze(dim=2)

        x = self.outer_positional_embedding(x)
        
        output_token = self.output_token.repeat(xeeg.shape[0], 1, 1, 1, 1, 1)

        x = self.outer_tf(output_token, x, extract_norm=extract_norm)

        return x
class SleepEnc_conv_ΕΕG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if "dim_proj" in args:
            dim_proj = args.dim_proj
        else:
            dim_proj = d_model
        if "num_layers" in args:
            num_layers = args.num_layers
        else:
            num_layers = 4

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf = inner_att_conv_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, dim_proj=dim_proj, num_layers=num_layers)

        self.outer_tf = outer_mod_att_conv_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, dim_proj=dim_proj, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf(xeeg, extract_norm=extract_norm)
        x = xeeg[:, :, 0].unsqueeze(dim=2)

        x = self.outer_positional_embedding(x)

        x = self.outer_tf(x, extract_norm=extract_norm)

        return x
class SleepEnc_TF_RN_ΕΕG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

        self.inner_rn = inner_rn_RA(d_model, inner=29, outer=21, modalities=1, num_layers=4)
        self.outer_rn = outer_mod_rn_RA(d_model, inner=29, outer=21, modalities=1, num_layers=4)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg_rn = self.inner_rn(xeeg)
        xeeg_rn = self.outer_rn(xeeg_rn)

        xeeg = self.inner_positional_embedding(xeeg)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf(xeeg, extract_norm=extract_norm)
        x = xeeg[:, :, 0].unsqueeze(dim=2)

        x = self.outer_positional_embedding(x)

        x = self.outer_tf(x, extract_norm=extract_norm)

        return torch.cat([xeeg_rn, x],dim=-1)
class SleepEnc_onlyinner_ΕΕG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf(xeeg, extract_norm=extract_norm)
        x = xeeg[:, :, 0].unsqueeze(dim=2)

        return x
class SleepEnc_onlyouter_ΕΕG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args
        print(args)

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, self.outer_cls, rpos = False, False, False, False
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "outer_cls" in args:
            self.outer_cls = args.outer_cls
        if "rpos" in args:
            rpos = args.rpos

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)


        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):

        x = self.outer_positional_embedding(x)
        if self.outer_cls:
            cls_token_eeg = self.cls_token.repeat(x.shape[0], 1, 1, 1, x.shape[3], 1)
            x = torch.cat([cls_token_eeg, x], dim=1)
        x = self.outer_tf(x, extract_norm=extract_norm)
        if self.outer_cls:
            x = x[:, 0].unsqueeze(dim=2)

        return x

class SleepEnc_ΕΕG_ConnEpoch(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_l0 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_l1 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_l2 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_l3 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.init_seq_cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.end_seq_cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        n_position = 21
        i_diagonal = n_position - 1
        diagonal_vals = torch.ones(i_diagonal, dtype=torch.long)
        self.diag_matrix_plus = nn.Parameter(torch.diagflat(diagonal_vals, offset=n_position - i_diagonal), requires_grad=False)
        self.diag_matrix_minus = nn.Parameter(torch.diagflat(diagonal_vals, offset=-n_position + i_diagonal), requires_grad=False)

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_l0(xeeg, extract_norm=extract_norm)
        x_cls = xeeg[:, :, :1]

        diag_matrix_plus = self.diag_matrix_plus[:x_cls.shape[1],:x_cls.shape[1]]
        diag_matrix_minus = self.diag_matrix_minus[:x_cls.shape[1],:x_cls.shape[1]]

        x_cls_neigh_minus = torch.einsum("mb,abcdgf->amcdgf", diag_matrix_minus[1:,:].float(), x_cls)
        x_cls_neigh_minus = torch.cat([self.init_seq_cls_token.repeat(x_cls.shape[0],1,1,1,1,1), x_cls_neigh_minus],dim=1)

        x_cls_neigh_plus = torch.einsum("mb,abcdgf->amcdgf", diag_matrix_plus[:-1,:].float(), x_cls)
        x_cls_neigh_plus = torch.cat([x_cls_neigh_plus, self.end_seq_cls_token.repeat(x_cls.shape[0],1,1,1,1,1)],dim=1)

        xeeg = torch.cat([xeeg, x_cls_neigh_minus, x_cls_neigh_plus], dim=2)
        xeeg = self.inner_tf_l1(xeeg, extract_norm=extract_norm)
        x_cls = xeeg[:, :, :1]

        x_cls_neigh_minus = torch.einsum("mb,abcdgf->amcdgf", diag_matrix_minus[1:,:].float(), x_cls)
        x_cls_neigh_minus = torch.cat([self.init_seq_cls_token.repeat(x_cls.shape[0],1,1,1,1,1), x_cls_neigh_minus],dim=1)

        x_cls_neigh_plus = torch.einsum("mb,abcdgf->amcdgf", diag_matrix_plus[:-1,:].float(), x_cls)
        x_cls_neigh_plus = torch.cat([x_cls_neigh_plus, self.end_seq_cls_token.repeat(x_cls.shape[0],1,1,1,1,1)],dim=1)
        xeeg = torch.cat([xeeg, x_cls_neigh_minus, x_cls_neigh_plus], dim=2)
        xeeg = self.inner_tf_l2(xeeg, extract_norm=extract_norm)
        x_cls = xeeg[:, :, :1]

        x_cls_neigh_minus = torch.einsum("mb,abcdgf->amcdgf", diag_matrix_minus[1:,:].float(), x_cls)
        x_cls_neigh_minus = torch.cat([self.init_seq_cls_token.repeat(x_cls.shape[0],1,1,1,1,1), x_cls_neigh_minus],dim=1)

        x_cls_neigh_plus = torch.einsum("mb,abcdgf->amcdgf", diag_matrix_plus[:-1,:].float(), x_cls)
        x_cls_neigh_plus = torch.cat([x_cls_neigh_plus, self.end_seq_cls_token.repeat(x_cls.shape[0],1,1,1,1,1)],dim=1)
        xeeg = torch.cat([xeeg, x_cls_neigh_minus, x_cls_neigh_plus], dim=2)
        xeeg = self.inner_tf_l3(xeeg, extract_norm=extract_norm)
        x = xeeg[:, :, :1]

        x = self.outer_positional_embedding(x)

        x = self.outer_tf(x, extract_norm=extract_norm)

        return x
class SleepEnc_ΕΕG_ConnEpoch(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_l0 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_l1 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_l2 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)
        self.inner_tf_l3 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=1)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.init_seq_cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))
        self.end_seq_cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        n_position = 21
        i_diagonal = n_position - 1
        diagonal_vals = torch.ones(i_diagonal, dtype=torch.long)
        self.diag_matrix_plus = nn.Parameter(torch.diagflat(diagonal_vals, offset=n_position - i_diagonal), requires_grad=False)
        self.diag_matrix_minus = nn.Parameter(torch.diagflat(diagonal_vals, offset=-n_position + i_diagonal), requires_grad=False)

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_l0(xeeg, extract_norm=extract_norm)
        x_cls = xeeg[:, :, :1]

        diag_matrix_plus = self.diag_matrix_plus[:x_cls.shape[1],:x_cls.shape[1]]
        diag_matrix_minus = self.diag_matrix_minus[:x_cls.shape[1],:x_cls.shape[1]]

        x_cls_neigh_minus = torch.einsum("mb,abcdgf->amcdgf", diag_matrix_minus[1:,:].float(), x_cls)
        x_cls_neigh_minus = torch.cat([self.init_seq_cls_token.repeat(x_cls.shape[0],1,1,1,1,1), x_cls_neigh_minus],dim=1)

        x_cls_neigh_plus = torch.einsum("mb,abcdgf->amcdgf", diag_matrix_plus[:-1,:].float(), x_cls)
        x_cls_neigh_plus = torch.cat([x_cls_neigh_plus, self.end_seq_cls_token.repeat(x_cls.shape[0],1,1,1,1,1)],dim=1)

        xeeg = torch.cat([xeeg, x_cls_neigh_minus, x_cls_neigh_plus], dim=2)
        xeeg = self.inner_tf_l1(xeeg, extract_norm=extract_norm)
        x_cls = xeeg[:, :, :1]

        x_cls_neigh_minus = torch.einsum("mb,abcdgf->amcdgf", diag_matrix_minus[1:,:].float(), x_cls)
        x_cls_neigh_minus = torch.cat([self.init_seq_cls_token.repeat(x_cls.shape[0],1,1,1,1,1), x_cls_neigh_minus],dim=1)

        x_cls_neigh_plus = torch.einsum("mb,abcdgf->amcdgf", diag_matrix_plus[:-1,:].float(), x_cls)
        x_cls_neigh_plus = torch.cat([x_cls_neigh_plus, self.end_seq_cls_token.repeat(x_cls.shape[0],1,1,1,1,1)],dim=1)
        xeeg = torch.cat([xeeg, x_cls_neigh_minus, x_cls_neigh_plus], dim=2)
        xeeg = self.inner_tf_l2(xeeg, extract_norm=extract_norm)
        x_cls = xeeg[:, :, :1]

        x_cls_neigh_minus = torch.einsum("mb,abcdgf->amcdgf", diag_matrix_minus[1:,:].float(), x_cls)
        x_cls_neigh_minus = torch.cat([self.init_seq_cls_token.repeat(x_cls.shape[0],1,1,1,1,1), x_cls_neigh_minus],dim=1)

        x_cls_neigh_plus = torch.einsum("mb,abcdgf->amcdgf", diag_matrix_plus[:-1,:].float(), x_cls)
        x_cls_neigh_plus = torch.cat([x_cls_neigh_plus, self.end_seq_cls_token.repeat(x_cls.shape[0],1,1,1,1,1)],dim=1)
        xeeg = torch.cat([xeeg, x_cls_neigh_minus, x_cls_neigh_plus], dim=2)
        xeeg = self.inner_tf_l3(xeeg, extract_norm=extract_norm)
        x = xeeg[:, :, :1]

        x = self.outer_positional_embedding(x)

        x = self.outer_tf(x, extract_norm=extract_norm)

        return x
class SleepEnc_ΕΕG_Multichannel(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf = inner_ch_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=inner_biased, num_layers=4)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)


        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3],  xeeg.shape[4], 1)

        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf(xeeg, extract_norm=extract_norm)

        x = xeeg[:, :, :1, :, :1]

        x = self.outer_positional_embedding(x)

        x = self.outer_tf(x, extract_norm=extract_norm)

        return x
class SleepEnc_Merged_glearnedbiasedplusouter_rpos_EEG_EOG(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        d_model = args.dmodel  # 64*8

        inner_biased, outer_biased, rpos = False, False, False
        if "inner_biased" in args:
            inner_biased = args.inner_biased
        if "outer_biased" in args:
            outer_biased = args.outer_biased
        if "rpos" in args:
            rpos = args.rpos

        if inner_biased == "gaussian_learned":
            inner_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        if outer_biased == "gaussian_learned":
            outer_biased = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf = inner_att_RA(d_model, pos=False, inner=29, outer=21, rpos=rpos, modalities=1, num_layers=4, gbiased=inner_biased)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, rpos=rpos, modalities=1, gbiased=outer_biased, num_layers=4)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, xeog], dim=2)
        xeeg = self.inner_tf(xeeg, extract_norm=extract_norm)
        x = xeeg[:, :, 0].unsqueeze(dim=2)

        x = self.outer_positional_embedding(x)

        x = self.outer_tf(x, extract_norm=extract_norm)

        return x


class EEG_SLEEP_Contrastive_contextproc_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel=0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model = 128  # 64*8

        rpos = False

        self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)

        self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)

        biased = False

        self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)

        self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)


        self.inner_tf_context_l0 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_context_l1 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_context_l2 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)

        self.outer_tf_context_l0 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_context_l1 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_context_l2 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)

        # self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased,
        #                                       num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_context_token = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.outer_context_token = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog, "b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        context_token = self.inner_context_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)

        cls_token_eeg = self.cls_token_mod0.repeat(xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, context_token], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :, :-1], xeeg[:, :, -1:]

        cls_token_eog = self.cls_token_mod1.repeat(xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog, context_token], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :, :-1], xeog[:, :, -1:]


        xtotal = torch.cat([context_token_xeeg, xeeg, xeog, context_token_xeog], dim=2)
        xtotal = self.inner_tf_context_l0(xtotal, extract_norm=extract_norm)
        context_token_xeeg, context_token_xeοg = xtotal[:, :, :1], xtotal[:, :, -1:]


        xeeg = torch.cat([cls_token_eeg, xeeg, context_token_xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :, :-1], xeeg[:, :, -1:]

        xeog = torch.cat([cls_token_eog, xeog, context_token_xeοg], dim=2)
        xeog = self.inner_tf_mod1_l1(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :, :-1], xeog[:, :, -1:]



        xtotal = torch.cat([context_token_xeeg, xeeg, xeog, context_token_xeog], dim=2)
        xtotal = self.inner_tf_context_l1(xtotal, extract_norm=extract_norm)
        context_token_xeeg, context_token_xeοg = xtotal[:, :, :1], xtotal[:, :, -1:]


        xeeg = torch.cat([cls_token_eeg, xeeg, context_token_xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :, :-1], xeeg[:, :, -1:]

        xeog = torch.cat([cls_token_eog, xeog, context_token_xeοg], dim=2)
        xeog = self.inner_tf_mod1_l2(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :, :-1], xeog[:, :, -1:]


        xtotal = torch.cat([context_token_xeeg, xeeg, xeog, context_token_xeog], dim=2)
        xtotal = self.inner_tf_context_l2(xtotal, extract_norm=extract_norm)
        context_token_xeeg, context_token_xeοg = xtotal[:, :, :1], xtotal[:, :, -1:]


        xeeg = torch.cat([cls_token_eeg, xeeg, context_token_xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :, :1], xeeg[:, :, -1:]

        xeog = torch.cat([cls_token_eog, xeog, context_token_xeοg], dim=2)
        xeog = self.inner_tf_mod1_l2(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :, :1], xeog[:, :, -1:]




        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        outer_context_token = self.inner_context_token.repeat(xeeg.shape[0], 1, 1, xeeg.shape[3], 1)

        xeeg = torch.cat([xeeg, outer_context_token], dim=1)
        xeeg = self.outer_tf_mod0_l0(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :-1], xeeg[:, -1:]

        xeog = torch.cat([xeog, outer_context_token], dim=1)
        xeog = self.outer_tf_mod1_l0(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :-1], xeog[:, -1:]

        xtotal = torch.cat([context_token_xeeg, xeeg, xeog, context_token_xeog], dim=1)
        xtotal = self.outer_tf_context_l0(xtotal, extract_norm=extract_norm)
        context_token_xeeg, context_token_xeοg = xtotal[:, :1], xtotal[:, -1:]

        xeeg = torch.cat([xeeg, context_token_xeeg], dim=1)
        xeeg = self.outer_tf_mod0_l1(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :-1], xeeg[:, -1:]

        xeog = torch.cat([xeog, context_token_xeοg], dim=1)
        xeog = self.outer_tf_mod1_l1(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :-1], xeog[:, -1:]

        xtotal = torch.cat([context_token_xeeg, xeeg, xeog, context_token_xeog], dim=1)
        xtotal = self.outer_tf_context_l1(xtotal, extract_norm=extract_norm)
        context_token_xeeg, context_token_xeοg = xtotal[:, :1], xtotal[:, -1:]

        xeeg = torch.cat([xeeg, context_token_xeeg], dim=1)
        xeeg = self.outer_tf_mod0_l2(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :-1], xeeg[:, -1:]

        xeog = torch.cat([xeog, context_token_xeοg], dim=1)
        xeog = self.outer_tf_mod1_l2(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :-1], xeog[:, -1:]

        xtotal = torch.cat([context_token_xeeg, xeeg, xeog, context_token_xeog], dim=1)
        xtotal = self.outer_tf_context_l2(xtotal, extract_norm=extract_norm)
        context_token_xeeg, context_token_xeοg = xtotal[:, :1], xtotal[:, -1:]

        xeeg = torch.cat([xeeg, context_token_xeeg], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :-1], xeeg[:, -1:]

        xeog = torch.cat([xeog, context_token_xeοg], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :-1], xeog[:, -1:]


        return xeeg, xeog
class EEG_SLEEP_Contrastive_contextproc_rpos_EEG_EOG(nn.Module):
    def __init__(self, args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model = 128  # 64*8

        rpos = True
        self.args = args

        self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)

        self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)

        biased = False

        self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)

        self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)


        self.inner_tf_context_l0 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_context_l1 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)
        self.inner_tf_context_l2 = inner_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, num_layers=1)

        self.outer_tf_context_l0 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_context_l1 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)
        self.outer_tf_context_l2 = outer_mod_att_RA(d_model, pos=False, rpos=rpos, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=1)

        # self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased,
        #                                       num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_context_token = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.outer_context_token = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog, "b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        context_token = self.inner_context_token.repeat(xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1).unsqueeze(dim=3)

        cls_token_eeg = self.cls_token_mod0.repeat(xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1).unsqueeze(dim=3)
        xeeg = torch.cat([cls_token_eeg, xeeg, context_token], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :, :-1], xeeg[:, :, -1:]

        cls_token_eog = self.cls_token_mod1.repeat(xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1).unsqueeze(dim=3)
        xeog = torch.cat([cls_token_eog, xeog, context_token], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :, :-1], xeog[:, :, -1:]


        xtotal = torch.cat([context_token_xeeg, xeeg, xeog, context_token_xeog], dim=2)
        xtotal = self.inner_tf_context_l0(xtotal, extract_norm=extract_norm)
        context_token_xeeg, context_token_xeοg = xtotal[:, :, :1], xtotal[:, :, -1:]


        xeeg = torch.cat([cls_token_eeg, xeeg, context_token_xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :, :-1], xeeg[:, :, -1:]

        xeog = torch.cat([cls_token_eog, xeog, context_token_xeοg], dim=2)
        xeog = self.inner_tf_mod1_l1(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :, :-1], xeog[:, :, -1:]



        xtotal = torch.cat([context_token_xeeg, xeeg, xeog, context_token_xeog], dim=2)
        xtotal = self.inner_tf_context_l1(xtotal, extract_norm=extract_norm)
        context_token_xeeg, context_token_xeοg = xtotal[:, :, :1], xtotal[:, :, -1:]


        xeeg = torch.cat([cls_token_eeg, xeeg, context_token_xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :, :-1], xeeg[:, :, -1:]

        xeog = torch.cat([cls_token_eog, xeog, context_token_xeοg], dim=2)
        xeog = self.inner_tf_mod1_l2(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :, :-1], xeog[:, :, -1:]


        xtotal = torch.cat([context_token_xeeg, xeeg, xeog, context_token_xeog], dim=2)
        xtotal = self.inner_tf_context_l2(xtotal, extract_norm=extract_norm)
        context_token_xeeg, context_token_xeοg = xtotal[:, :, :1], xtotal[:, :, -1:]


        xeeg = torch.cat([cls_token_eeg, xeeg, context_token_xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :, :1], xeeg[:, :, -1:]

        xeog = torch.cat([cls_token_eog, xeog, context_token_xeοg], dim=2)
        xeog = self.inner_tf_mod1_l2(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :, :1], xeog[:, :, -1:]




        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        outer_context_token = self.inner_context_token.repeat(xeeg.shape[0], 1, 1, xeeg.shape[3], 1).unsqueeze(dim=3)

        xeeg = torch.cat([xeeg, outer_context_token], dim=1)
        xeeg = self.outer_tf_mod0_l0(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :-1], xeeg[:, -1:]

        xeog = torch.cat([xeog, outer_context_token], dim=1)
        xeog = self.outer_tf_mod1_l0(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :-1], xeog[:, -1:]

        xtotal = torch.cat([context_token_xeeg, xeeg, xeog, context_token_xeog], dim=1)
        xtotal = self.outer_tf_context_l0(xtotal, extract_norm=extract_norm)
        context_token_xeeg, context_token_xeοg = xtotal[:, :1], xtotal[:, -1:]

        xeeg = torch.cat([xeeg, context_token_xeeg], dim=1)
        xeeg = self.outer_tf_mod0_l1(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :-1], xeeg[:, -1:]

        xeog = torch.cat([xeog, context_token_xeοg], dim=1)
        xeog = self.outer_tf_mod1_l1(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :-1], xeog[:, -1:]

        xtotal = torch.cat([context_token_xeeg, xeeg, xeog, context_token_xeog], dim=1)
        xtotal = self.outer_tf_context_l1(xtotal, extract_norm=extract_norm)
        context_token_xeeg, context_token_xeοg = xtotal[:, :1], xtotal[:, -1:]

        xeeg = torch.cat([xeeg, context_token_xeeg], dim=1)
        xeeg = self.outer_tf_mod0_l2(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :-1], xeeg[:, -1:]

        xeog = torch.cat([xeog, context_token_xeοg], dim=1)
        xeog = self.outer_tf_mod1_l2(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :-1], xeog[:, -1:]

        xtotal = torch.cat([context_token_xeeg, xeeg, xeog, context_token_xeog], dim=1)
        xtotal = self.outer_tf_context_l2(xtotal, extract_norm=extract_norm)
        context_token_xeeg, context_token_xeοg = xtotal[:, :1], xtotal[:, -1:]

        xeeg = torch.cat([xeeg, context_token_xeeg], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg, extract_norm=extract_norm)
        xeeg, context_token_xeeg = xeeg[:, :-1], xeeg[:, -1:]

        xeog = torch.cat([xeog, context_token_xeοg], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog, extract_norm=extract_norm)
        xeog, context_token_xeog = xeog[:, :-1], xeog[:, -1:]


        return xeeg, xeog

class EEG_SLEEP_Contrastive_big_moddrop_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel=0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model = 128  # 64*8

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, num_layers=16)

        self.inner_tf_mod1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, num_layers=16)

        biased = False

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased,
                                              num_layers=16)

        self.outer_tf_mod1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased,
                                              num_layers=16)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

        self.mod_dropout_mod1 = Multimodal_Dropout_outer(d_model, dropout_prob=0.15)
        self.mod_dropout_mod0 = Multimodal_Dropout_outer(d_model, dropout_prob=0.15)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog, "b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        cls_token_eeg = self.cls_token_mod0.repeat(xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat(xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)
        xeog = self.inner_tf_mod1(xeog, extract_norm=extract_norm)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        xeeg = self.mod_dropout_mod0(xeeg)
        xeog = self.mod_dropout_mod1(xeog)

        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)
        xeog = self.outer_tf_mod1(xeog, extract_norm=extract_norm)

        return xeeg, xeog
class EEG_SLEEP_Contrastive_rpos_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel=0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model = 128  # 64*8

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        self.inner_tf_mod1 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        biased = False

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, gbiased=biased,
                                              num_layers=4)

        self.outer_tf_mod1 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, gbiased=biased,
                                              num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                          channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1,
                                                        channels=1)

    def forward(self, x, inits=None, extract_norm=False):
        xeeg = x[0][:, :, :, 1:, :]  # mat
        xeog = x[1][:, :, :, 1:, :]  # mat

        xeeg = einops.rearrange(xeeg, "b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog, "b outer mod f inner -> b outer inner mod f")

        # xeeg = self.inner_positional_embedding_0(xeeg)
        # xeog = self.inner_positional_embedding_1(xeog)

        cls_token_eeg = self.cls_token_mod0.repeat(xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat(xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog], dim=2)
        xeog = self.inner_tf_mod1(xeog, extract_norm=extract_norm)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)

        # xeeg = self.outer_positional_embedding(xeeg)
        # xeog = self.outer_positional_embedding(xeog)

        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)
        xeog = self.outer_tf_mod1(xeog, extract_norm=extract_norm)

        return xeeg, xeog
class EEG_SLEEP_Contrastive_gbiased_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        gbiased = Gaussian_Attention_Bias(rate=1)

        self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)

        self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)


        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)



        xeog = self.inner_tf_mod1_l0(xeog)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog = self.inner_tf_mod1_l2(xeog)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        xeeg = self.outer_tf_mod0_l0(xeeg, extract_norm=extract_norm)
        xeeg = self.outer_tf_mod0_l1(xeeg, extract_norm=extract_norm)
        xeeg = self.outer_tf_mod0_l2(xeeg, extract_norm=extract_norm)
        xeeg = self.outer_tf_mod0_l3(xeeg, extract_norm=extract_norm)

        xeog = self.outer_tf_mod1_l0(xeog, extract_norm=extract_norm)
        xeog = self.outer_tf_mod1_l1(xeog, extract_norm=extract_norm)
        xeog = self.outer_tf_mod1_l2(xeog, extract_norm=extract_norm)
        xeog = self.outer_tf_mod1_l3(xeog, extract_norm=extract_norm)

        return xeeg, xeog
class EEG_SLEEP_Contrastive_gbiasedm_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        gbiased = Gaussian_Attention_Bias()

        self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)

        self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)


        xeeg = self.inner_tf_mod0_l0(xeeg, extract_norm=extract_norm)
        xeeg = self.inner_tf_mod0_l1(xeeg, extract_norm=extract_norm)
        xeeg = self.inner_tf_mod0_l2(xeeg, extract_norm=extract_norm)
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)



        xeog = self.inner_tf_mod1_l0(xeog, extract_norm=extract_norm)
        xeog = self.inner_tf_mod1_l1(xeog, extract_norm=extract_norm)
        xeog = self.inner_tf_mod1_l2(xeog, extract_norm=extract_norm)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog, extract_norm=extract_norm)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        xeeg = self.outer_tf_mod0_l0(xeeg, extract_norm=extract_norm)
        xeeg = self.outer_tf_mod0_l1(xeeg, extract_norm=extract_norm)
        xeeg = self.outer_tf_mod0_l2(xeeg, extract_norm=extract_norm)
        xeeg = self.outer_tf_mod0_l3(xeeg, extract_norm=extract_norm)

        xeog = self.outer_tf_mod1_l0(xeog, extract_norm=extract_norm)
        xeog = self.outer_tf_mod1_l1(xeog, extract_norm=extract_norm)
        xeog = self.outer_tf_mod1_l2(xeog, extract_norm=extract_norm)
        xeog = self.outer_tf_mod1_l3(xeog, extract_norm=extract_norm)

        return xeeg, xeog

class EEG_SLEEP_Contrastive_gbiasedm_plus_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        biased =  Gaussian_Attention_Bias(type="add")

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        self.inner_tf_mod1 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased, num_layers=4)

        self.outer_tf_mod1 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased, num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1(xeog, extract_norm=extract_norm)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        xeog = self.outer_tf_mod1(xeog, extract_norm=extract_norm)

        return xeeg, xeog
class EEG_SLEEP_Contrastive_glearnedbiasedm_plus_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        biased_inner_mod_0 = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")
        biased_inner_mod_1 = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")
        biased_outer_mod_0 = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")
        biased_outer_mod_1 = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased_inner_mod_0, num_layers=4)

        self.inner_tf_mod1 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased_inner_mod_1, num_layers=4)

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased_outer_mod_0, num_layers=4)

        self.outer_tf_mod1 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased_outer_mod_1, num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1(xeog, extract_norm=extract_norm)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        xeog = self.outer_tf_mod1(xeog, extract_norm=extract_norm)

        return xeeg, xeog

class EEG_SLEEP_Contrastive_lstm_rpos_EEG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        # self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        self.lstm = outer_mod_lstm(d_model, inner=29, outer=21, modalities=1, num_layers=2)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding(xeeg)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)
        #
        # xeeg = self.outer_positional_embedding(xeeg)
        #
        # xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        xeeg = self.lstm(xeeg)

        return xeeg
class EEG_SLEEP_Contrastive_normreg20_rpos_EEG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0 = inner_att_normreg_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        self.outer_tf_mod0 = outer_mod_att_normreg_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        # self.lstm = outer_mod_lstm(d_model, inner=29, outer=21, modalities=1, num_layers=2)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding(xeeg)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)
        #
        xeeg = self.outer_positional_embedding(xeeg)
        #
        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        # xeeg = self.lstm(xeeg)

        return xeeg
class EEG_SLEEP_Contrastive_rpos_time2vec_EEG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_mod0 = inner_att_RA(d_model, pos=False, rpos=False, inner=29, outer=21, modalities=1, num_layers=4)

        self.outer_mod0 = outer_mod_att_RA(d_model, pos=False, rpos=False, inner=29, outer=21, modalities=1, num_layers=4)

        # self.lstm = outer_mod_lstm(d_model, inner=29, outer=21, modalities=1, num_layers=2)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(128, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

        self.w = nn.Parameter(torch.hamming_window(200), requires_grad=False)

        self.omegas = nn.Parameter(torch.rand([127,1,200]), requires_grad=True)
        self.fis = nn.Parameter(torch.rand([127,1]), requires_grad=True)

        self.omegas_0 = nn.Parameter(torch.rand([1,1,200]), requires_grad=True)
        self.fis_0 = nn.Parameter(torch.rand([1,1]), requires_grad=True)

    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat

        xeeg_time = x[1]#mat
        xeeg_time = xeeg_time.unfold(-1,200,100)*self.w
        xeeg_time_1 = einsum("abcd, kld -> abckl", xeeg_time, self.omegas)
        xeeg_time_1 = torch.sin((xeeg_time_1 + self.fis).squeeze()).unsqueeze(dim=-2)

        xeeg_time_0 = einsum("abcd, kld -> abckl", xeeg_time, self.omegas_0)
        xeeg_time_0 = torch.sin((xeeg_time_0 + self.fis_0).squeeze(dim=-1)).unsqueeze(dim=-2)

        xeeg_time = torch.cat([xeeg_time_0, xeeg_time_1], dim=-1)
        xeeg_time = einops.rearrange(xeeg_time,"b outer inner mod f -> b outer inner mod f")


        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeeg = xeeg + xeeg_time
        # xeeg = self.inner_positional_embedding(xeeg)

        # xeeg = torch.cat([xeeg,xeeg_time], dim=-1)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)

        xeeg = self.inner_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)
        #
        # xeeg = self.outer_positional_embedding(xeeg)
        #
        xeeg = self.outer_mod0(xeeg, extract_norm=extract_norm)

        return xeeg
class EEG_SLEEP_Contrastive_intlstm_rpos_EEG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        self.outer_mod0 = outer_mod_intlstm(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        # self.lstm = outer_mod_intlstm(d_model, inner=29, outer=21, modalities=1, num_layers=2)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding(xeeg)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)
        #
        # xeeg = self.outer_positional_embedding(xeeg)
        #
        # xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        xeeg = self.outer_mod0(xeeg)

        return xeeg
class EEG_SLEEP_Contrastive_io_intlstm_rpos_EEG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_mod0 = inner_intlstm(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        self.outer_mod0 = outer_mod_intlstm(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        # self.lstm = outer_mod_intlstm(d_model, inner=29, outer=21, modalities=1, num_layers=2)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding(xeeg)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([xeeg, cls_token_eeg], dim=2)
        xeeg = self.inner_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, -1].unsqueeze(dim=2)
        #
        xeeg = self.outer_positional_embedding(xeeg)
        #
        # xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        xeeg = self.outer_mod0(xeeg)

        return xeeg
class EEG_SLEEP_Contrastive_rpos_glearnedbiasedm_outer_plus_EEG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        biased_outer_mod_0 = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased_outer_mod_0, num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding(xeeg)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)

        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        return xeeg
class EEG_SLEEP_Contrastive_rpos_glearnedbiasednodiagm_outer_plus_EEG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        biased_outer_mod_0 = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul", with_diag=False)

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased_outer_mod_0, num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding(xeeg)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)

        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        return xeeg

class EEG_SLEEP_Contrastive_glearnedbiasedm_outerplus_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], args=None):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        self.inner_tf_mod1 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        biased_outer_mod_0 = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")
        biased_outer_mod_1 = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased_outer_mod_0, num_layers=4)

        self.outer_tf_mod1 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased_outer_mod_1, num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1) 



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod ch f inner -> b outer inner mod ch f")
        xeog = einops.rearrange(xeog,"b outer mod ch f inner -> b outer inner mod ch f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1(xeog, extract_norm=extract_norm)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        xeog = self.outer_tf_mod1(xeog, extract_norm=extract_norm)

        return xeeg, xeog
class EEG_SLEEP_Contrastive_glearnedbiasedm_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, num_layers=4)

        self.inner_tf_mod1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, num_layers=4)

        biased_outer_mod_0 = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")
        biased_outer_mod_1 = Gaussian_Learned_Attention_Bias(d_model, heads=8, type="mul")

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased_outer_mod_0, num_layers=4)

        self.outer_tf_mod1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased_outer_mod_1, num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)


        # xeeg = self.inner_tf_mod0_l0(xeeg)
        # xeeg = self.inner_tf_mod0_l1(xeeg)
        # xeeg = self.inner_tf_mod0_l2(xeeg)
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)



        # xeog = self.inner_tf_mod1_l0(xeog)
        # xeog = self.inner_tf_mod1_l1(xeog)
        # xeog = self.inner_tf_mod1_l2(xeog)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1(xeog, extract_norm=extract_norm)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        # xeeg = self.outer_tf_mod0_l0(xeeg, extract_norm=extract_norm)
        # xeeg = self.outer_tf_mod0_l1(xeeg, extract_norm=extract_norm)
        # xeeg = self.outer_tf_mod0_l2(xeeg, extract_norm=extract_norm)
        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        # xeog = self.outer_tf_mod1_l0(xeog, extract_norm=extract_norm)
        # xeog = self.outer_tf_mod1_l1(xeog, extract_norm=extract_norm)
        # xeog = self.outer_tf_mod1_l2(xeog, extract_norm=extract_norm)
        xeog = self.outer_tf_mod1(xeog, extract_norm=extract_norm)

        return xeeg, xeog

class EEG_SLEEP_Contrastive_neighbiasedm3_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        # self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, num_layers=4)

        # self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, num_layers=4)

        biased = Attention_Bias_Neigh(num_diagonals=3, type="mul")

        # self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased)
        # self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased)
        # self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased)
        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=4)

        # self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased)
        # self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased)
        # self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased)
        self.outer_tf_mod1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=4)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)


        # xeeg = self.inner_tf_mod0_l0(xeeg)
        # xeeg = self.inner_tf_mod0_l1(xeeg)
        # xeeg = self.inner_tf_mod0_l2(xeeg)
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)



        # xeog = self.inner_tf_mod1_l0(xeog)
        # xeog = self.inner_tf_mod1_l1(xeog)
        # xeog = self.inner_tf_mod1_l2(xeog)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1(xeog, extract_norm=extract_norm)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        # xeeg = self.outer_tf_mod0_l0(xeeg, extract_norm=extract_norm)
        # xeeg = self.outer_tf_mod0_l1(xeeg, extract_norm=extract_norm)
        # xeeg = self.outer_tf_mod0_l2(xeeg, extract_norm=extract_norm)
        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        # xeog = self.outer_tf_mod1_l0(xeog, extract_norm=extract_norm)
        # xeog = self.outer_tf_mod1_l1(xeog, extract_norm=extract_norm)
        # xeog = self.outer_tf_mod1_l2(xeog, extract_norm=extract_norm)
        xeog = self.outer_tf_mod1(xeog, extract_norm=extract_norm)

        return xeeg, xeog
class EEG_SLEEP_Contrastive_neighbiasedm5_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        # self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, num_layers=4)

        # self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, num_layers=4)

        biased = Attention_Bias_Neigh(num_diagonals=5, type="mul")

        # self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased)
        # self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased)
        # self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased)
        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=4)

        # self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased)
        # self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased)
        # self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased)
        self.outer_tf_mod1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=biased, num_layers=4)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)


        # xeeg = self.inner_tf_mod0_l0(xeeg)
        # xeeg = self.inner_tf_mod0_l1(xeeg)
        # xeeg = self.inner_tf_mod0_l2(xeeg)
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)



        # xeog = self.inner_tf_mod1_l0(xeog)
        # xeog = self.inner_tf_mod1_l1(xeog)
        # xeog = self.inner_tf_mod1_l2(xeog)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1(xeog, extract_norm=extract_norm)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        # xeeg = self.outer_tf_mod0_l0(xeeg, extract_norm=extract_norm)
        # xeeg = self.outer_tf_mod0_l1(xeeg, extract_norm=extract_norm)
        # xeeg = self.outer_tf_mod0_l2(xeeg, extract_norm=extract_norm)
        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        # xeog = self.outer_tf_mod1_l0(xeog, extract_norm=extract_norm)
        # xeog = self.outer_tf_mod1_l1(xeog, extract_norm=extract_norm)
        # xeog = self.outer_tf_mod1_l2(xeog, extract_norm=extract_norm)
        xeog = self.outer_tf_mod1(xeog, extract_norm=extract_norm)

        return xeeg, xeog
class EEG_SLEEP_Contrastive_neighbiasedm3_plus_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        biased = Attention_Bias_Neigh(num_diagonals=3, type = "mul")

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased, num_layers=4)

        self.inner_tf_mod1 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased, num_layers=4)

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased, num_layers=4)

        self.outer_tf_mod1 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased, num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1(xeog, extract_norm=extract_norm)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        xeog = self.outer_tf_mod1(xeog, extract_norm=extract_norm)

        return xeeg, xeog
class EEG_SLEEP_Contrastive_neighbiasedm5_plus_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        biased = Attention_Bias_Neigh(num_diagonals=5, type = "mul")

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased, num_layers=4)

        self.inner_tf_mod1 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased, num_layers=4)

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased, num_layers=4)

        self.outer_tf_mod1 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased, num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1(xeog, extract_norm=extract_norm)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        xeog = self.outer_tf_mod1(xeog, extract_norm=extract_norm)

        return xeeg, xeog
class EEG_SLEEP_Contrastive_neighbiasedm5_outerplus_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        self.inner_tf_mod1 = inner_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, num_layers=4)

        biased = Attention_Bias_Neigh(num_diagonals=5, type = "mul")

        self.outer_tf_mod0 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased, num_layers=4)

        self.outer_tf_mod1 = outer_mod_att_RA(d_model, pos=False, rpos=True, inner=29, outer=21, modalities=1, extra_attention=biased, num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0(xeeg, extract_norm=extract_norm)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1(xeog, extract_norm=extract_norm)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        xeeg = self.outer_tf_mod0(xeeg, extract_norm=extract_norm)

        xeog = self.outer_tf_mod1(xeog, extract_norm=extract_norm)

        return xeeg, xeog


class EEG_SLEEP_Contrastive_gbiased_diag_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        gbiased = Gaussian_Attention_Bias(rate=1, with_diagonal=True)

        self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)

        self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)


        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)



        xeog = self.inner_tf_mod1_l0(xeog)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog = self.inner_tf_mod1_l2(xeog)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        xeeg = self.outer_tf_mod0_l0(xeeg, extract_norm=extract_norm)
        xeeg = self.outer_tf_mod0_l1(xeeg, extract_norm=extract_norm)
        xeeg = self.outer_tf_mod0_l2(xeeg, extract_norm=extract_norm)
        xeeg = self.outer_tf_mod0_l3(xeeg, extract_norm=extract_norm)

        xeog = self.outer_tf_mod1_l0(xeog, extract_norm=extract_norm)
        xeog = self.outer_tf_mod1_l1(xeog, extract_norm=extract_norm)
        xeog = self.outer_tf_mod1_l2(xeog, extract_norm=extract_norm)
        xeog = self.outer_tf_mod1_l3(xeog, extract_norm=extract_norm)

        return xeeg, xeog
class EEG_SLEEP_Contrastive_rep_bottleneck_gbiased_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        gbiased = Gaussian_Attention_Bias(rate=0.2, with_diagonal=False)

        self.common_tf_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.common_tf_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.common_tf_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.common_tf_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)

        self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)

        self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_common = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 7, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        common_outer_expansion = int(xeeg.shape[1]/ self.common_bottleneck.shape[1])
        common =  self.common_bottleneck.repeat( xeeg.shape[0], common_outer_expansion, 1, 1, 1)

        common = self.repetitive_forward(common, xeeg, xeog, extract_norm=extract_norm)
        common = self.repetitive_forward(common, xeeg, xeog, extract_norm=extract_norm)
        common = self.repetitive_forward(common, xeeg, xeog, extract_norm=extract_norm)

        return common

    def repetitive_forward(self, common, xeeg, xeog, extract_norm=False):

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        common = self.cls_token_common.repeat( common.shape[0], common.shape[1], 5, 1, 1)

        # Inner layer 0
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -5:]

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, 1, 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -5:]

        common = self.common_tf_l0(common)
        cls_token_common = self.cls_token_common.repeat( common.shape[0], common.shape[1], 1, 1, 1)
        common = torch.cat([cls_token_common, common],dim=2)
        common = self.common_tf_l1(common)
        common = common[:, :, 0].unsqueeze(dim=2)

        common = self.common_tf_l2(common)
        common = self.common_tf_l3(common)

        # common = common[:, :, 0].unsqueeze(dim=2)

        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l0(xeeg, extract_norm=extract_norm)
        xeeg, common = xeeg[:,:-21], xeeg[:,-21:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l0(xeog, extract_norm=extract_norm)
        xeog, common = xeeg[:,:-21], xeog[:,-21:]

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l1(xeeg, extract_norm=extract_norm)
        xeeg, common = xeeg[:,:-21], xeeg[:,-21:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l1(xeog, extract_norm=extract_norm)
        xeog, common = xeeg[:,:-21], xeog[:,-21:]

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l2(xeeg, extract_norm=extract_norm)
        xeeg, common = xeeg[:,:-21], xeeg[:,-21:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l2(xeog, extract_norm=extract_norm)
        xeog, common = xeeg[:,:-21], xeog[:,-21:]

        xeeg = torch.cat([xeeg, common], dim=1)
        xeeg = self.outer_tf_mod0_l3(xeeg, extract_norm=extract_norm)
        xeeg, common = xeeg[:,:-21], xeeg[:,-21:]

        xeog = torch.cat([xeog, common], dim=1)
        xeog = self.outer_tf_mod1_l3(xeog, extract_norm=extract_norm)
        xeog, common = xeeg[:,:-21], xeog[:,-21:]

        return common
class EEG_SLEEP_Contrastive_rep_bottleneck_2c_gbiased_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        # gbiased = Gaussian_Attention_Bias(rate=0.2, with_diagonal=False)

        self.common_tf_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.common_tf_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.common_tf_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.common_tf_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.common_tf_l4 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.common_tf_l5 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        # self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        # self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        # self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        # self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        #
        # self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        # self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        # self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)
        # self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_common = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.cls_token_common_2 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 7, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 7, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        common_outer_expansion = int(xeeg.shape[1]/ self.common_bottleneck.shape[1])
        common =  self.common_bottleneck.repeat( xeeg.shape[0], common_outer_expansion, 1, 1, 1)

        common = self.repetitive_forward(common, xeeg, xeog, extract_norm=extract_norm)
        common = self.repetitive_forward(common, xeeg, xeog, extract_norm=extract_norm)
        common = self.repetitive_forward(common, xeeg, xeog, extract_norm=extract_norm)

        return common

    def repetitive_forward(self, common, xeeg, xeog, extract_norm=False):

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        common = self.cls_token_common.repeat( common.shape[0], common.shape[1], 5, 1, 1)

        # Inner layer 0
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod1_l0(xeog)
        xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l2(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -5:]

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, 1, 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -5:]

        common = self.common_tf_l0(common)
        cls_token_common = self.cls_token_common.repeat( common.shape[0], common.shape[1], 1, 1, 1)
        common = torch.cat([cls_token_common, common],dim=2)
        common = self.common_tf_l1(common)
        common = common[:, :, 0].unsqueeze(dim=2)

        # common = torch.cat([xeeg, common, xeog], dim=-1)
        common = self.outer_positional_embedding(common)

        common = self.common_tf_l2(common)
        common = self.common_tf_l3(common)
        common = self.common_tf_l4(common)
        common = self.common_tf_l5(common)

        # common = common[:, :, 0].unsqueeze(dim=2)

        return common
class EEG_SLEEP_Contrastive_merged_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, num_layers=8)

        gbiased = Gaussian_Attention_Bias(rate=0.2, with_diagonal=False)

        self.outer_tf = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=gbiased, num_layers=8)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        x = torch.cat([xeeg, xeog], dim=2)
        x = self.inner_positional_embedding(x)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, 1, 1)
        x = torch.cat([cls_token_eeg, cls_token_eog, x], dim=2)
        x = self.inner_tf(x, extract_norm=extract_norm)
        xeeg, xeog = x[:,:,0:1], x[:,:,1:2]
        x = torch.cat([xeeg, xeog], dim=1)
        x = self.outer_tf(x, extract_norm=extract_norm)
        xeeg, xeog = x[:,:xeeg.shape[1]], x[:,xeeg.shape[1]:]
        x = torch.cat([xeeg, xeog], dim=-1)

        return x
class EEG_SLEEP_Contrastive_merged_fc_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf = inner_att_fc_RA(d_model, pos=False, inner=29, outer=21, modalities=2, num_layers=4)

        # gbiased = Gaussian_Attention_Bias(rate=0.2, with_diagonal=False)

        self.outer_tf = outer_mod_att_fc_RA(d_model, pos=False, inner=29, outer=21, modalities=1, gbiased=False, num_layers=4)

        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)

    def forward(self, x, inits= None, extract_norm=False):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        x = torch.cat([xeeg, xeog], dim=2)
        x = self.inner_positional_embedding(x)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, 1, 1)
        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, 1, 1)

        x = torch.cat([cls_token_eeg, x, cls_token_eog], dim=2)
        x = self.inner_tf(x, extract_norm=extract_norm)
        xeeg, xeog = x[:,:,0:1], x[:,:,-1:]
        x = torch.cat([xeeg, xeog], dim=1)
        x = self.outer_tf(x, extract_norm=extract_norm)
        xeeg, xeog = x[:,:xeeg.shape[1]], x[:,xeeg.shape[1]:]
        x = torch.cat([xeeg, xeog], dim=-1)

        return x


class EEG_SLEEP_Contrastive_every_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)


        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg = self.inner_positional_embedding(xeeg)

        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg = self.inner_positional_embedding(xeeg)

        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg = self.inner_positional_embedding(xeeg)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)



        xeog = self.inner_tf_mod1_l0(xeog)
        xeog = self.inner_positional_embedding(xeog)

        xeog = self.inner_tf_mod1_l1(xeog)
        xeog = self.inner_positional_embedding(xeog)

        xeog = self.inner_tf_mod1_l2(xeog)
        xeog = self.inner_positional_embedding(xeog)


        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        xeeg = self.outer_tf_mod0_l0(xeeg)
        xeeg = self.outer_positional_embedding(xeeg)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg = self.outer_positional_embedding(xeeg)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg = self.outer_positional_embedding(xeeg)
        xeeg = self.outer_tf_mod0_l3(xeeg)


        xeog = self.outer_tf_mod1_l0(xeog)
        xeog = self.outer_positional_embedding(xeog)

        xeog = self.outer_tf_mod1_l1(xeog)
        xeog = self.outer_positional_embedding(xeog)

        xeog = self.outer_tf_mod1_l2(xeog)
        xeog = self.outer_positional_embedding(xeog)
        xeog = self.outer_tf_mod1_l3(xeog)

        return xeeg, xeog
class EEG_SLEEP_Contrastive_norm_RA_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_mod_att_Norm_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_mod_att_Norm_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_mod_att_Norm_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_mod_att_Norm_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_mod_att_Norm_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_mod_att_Norm_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_mod_att_Norm_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_mod_att_Norm_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding(xeeg)
        xeog = self.inner_positional_embedding(xeog)


        xeeg = self.inner_tf_mod0_l0(xeeg)

        xeeg = self.inner_tf_mod0_l1(xeeg)

        xeeg = self.inner_tf_mod0_l2(xeeg)

        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)



        xeog = self.inner_tf_mod1_l0(xeog)

        xeog = self.inner_tf_mod1_l1(xeog)

        xeog = self.inner_tf_mod1_l2(xeog)


        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding(xeeg)
        xeog = self.outer_positional_embedding(xeog)

        xeeg = self.outer_tf_mod0_l0(xeeg)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg = self.outer_tf_mod0_l3(xeeg)


        xeog = self.outer_tf_mod1_l0(xeog)
        xeog = self.outer_tf_mod1_l1(xeog)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog = self.outer_tf_mod1_l3(xeog)

        return xeeg, xeog
class EEG_SLEEP_Contrastive_HF_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        bert_configs = BertConfig()
        bert_configs.hidden_size = d_model
        bert_configs.intermediate_size = d_model
        bert_configs.num_attention_heads = 8
        bert_configs.num_hidden_layers = 4

        print(bert_configs)

        self.inner_tf_mod0 = BertModel(bert_configs)
        self.inner_tf_mod1 = BertModel(bert_configs)

        self.outer_tf_mod0 = BertModel(bert_configs)
        self.outer_tf_mod1 = BertModel(bert_configs)

    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        b, outer = xeeg.shape[0],xeeg.shape[1]

        xeeg = einops.rearrange(xeeg,"b outer inner mod f -> (b outer) inner (mod f)")
        xeog = einops.rearrange(xeog,"b outer inner mod f -> (b outer) inner (mod f)")

        xeeg = self.inner_tf_mod0(inputs_embeds=xeeg, output_hidden_states=True, output_attentions=True, output_norms=True)
        xeog = self.inner_tf_mod1(inputs_embeds=xeog, output_hidden_states=True, output_attentions=True, output_norms=True)

        # last_hidden_state, pooler_output, hidden_states, attentions, norms
        xeeg = xeeg[1]
        xeog = xeog[1]

        xeeg = einops.rearrange(xeeg,"(b outer) f -> b outer f", b=b, outer=outer)
        xeog = einops.rearrange(xeog,"(b outer) f -> b outer f", b=b, outer=outer)

        xeeg = self.outer_tf_mod0(inputs_embeds=xeeg, output_hidden_states=True, output_attentions=True, output_norms=True)
        xeog = self.outer_tf_mod1(inputs_embeds=xeog, output_hidden_states=True, output_attentions=True, output_norms=True)

        xeeg = xeeg[0]
        xeog = xeog[0]

        return xeeg, xeog
class EEG_SLEEP_Contrastive_merged_HF_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8
        bert_configs = BertConfig()
        bert_configs.hidden_size = d_model
        bert_configs.intermediate_size = d_model
        bert_configs.num_attention_heads = 8
        bert_configs.num_hidden_layers = 4

        print(bert_configs)
        self.inner_tf = BertModel(bert_configs)

        self.outer_tf = BertModel(bert_configs)

    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        x = torch.cat([xeeg,xeog],dim=2)

        b, outer = x.shape[0],x.shape[1]

        x = einops.rearrange(x,"b outer inner mod f -> (b outer) inner (mod f)")

        x = self.inner_tf(inputs_embeds=x, output_hidden_states=True, output_attentions=True, output_norms=True)

        # last_hidden_state, pooler_output, hidden_states, attentions, norms
        x = x[1]

        x = einops.rearrange(x,"(b outer) f -> b outer f", b=b, outer=outer)

        x = self.outer_tf(inputs_embeds=x, output_hidden_states=True, output_attentions=True, output_norms=True)

        x = x[0]

        return x

class EEG_SLEEP_Contrastive_learnable_pos_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = learnable_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = learnable_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding_0 = learnable_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding_1 = learnable_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)


        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)



        xeog = self.inner_tf_mod1_l0(xeog)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog = self.inner_tf_mod1_l2(xeog)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding_0(xeeg)
        xeog = self.outer_positional_embedding_1(xeog)

        xeeg = self.outer_tf_mod0_l0(xeeg)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg = self.outer_tf_mod0_l3(xeeg)

        xeog = self.outer_tf_mod1_l0(xeog)
        xeog = self.outer_tf_mod1_l1(xeog)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog = self.outer_tf_mod1_l3(xeog)

        return xeeg, xeog
class EEG_SLEEP_Contrastive_no_pos_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)

        xeog = self.inner_tf_mod1_l0(xeog)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog = self.inner_tf_mod1_l2(xeog)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)

        xeeg = self.outer_tf_mod0_l0(xeeg)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg = self.outer_tf_mod0_l3(xeeg)

        xeog = self.outer_tf_mod1_l0(xeog)
        xeog = self.outer_tf_mod1_l1(xeog)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog = self.outer_tf_mod1_l3(xeog)

        return xeeg, xeog
class EEG_SLEEP_Contrastive_concat_pos_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_mod_att_RA(d_model*2, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_mod_att_RA(d_model*2, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_mod_att_RA(d_model*2, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_mod_att_RA(d_model*2, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod1_l0 = outer_mod_att_RA(d_model*2, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l1 = outer_mod_att_RA(d_model*2, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l2 = outer_mod_att_RA(d_model*2, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod1_l3 = outer_mod_att_RA(d_model*2, pos=False, inner=29, outer=21, modalities=1)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding_0 = huy_pos_concat_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding_1 = huy_pos_concat_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)


        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)


        xeog = self.inner_tf_mod1_l0(xeog)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog = self.inner_tf_mod1_l2(xeog)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        xeeg = self.outer_positional_embedding_0(xeeg)
        xeog = self.outer_positional_embedding_1(xeog)

        xeeg = self.outer_tf_mod0_l0(xeeg)
        xeeg = self.outer_tf_mod0_l1(xeeg)
        xeeg = self.outer_tf_mod0_l2(xeeg)
        xeeg = self.outer_tf_mod0_l3(xeeg)

        xeog = self.outer_tf_mod1_l0(xeog)
        xeog = self.outer_tf_mod1_l1(xeog)
        xeog = self.outer_tf_mod1_l2(xeog)
        xeog = self.outer_tf_mod1_l3(xeog)

        return xeeg, xeog

class EEG_SLEEP_Contrastive_only_inner_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()

        d_model =  128#64*8

        self.inner_tf_mod0_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.inner_tf_mod1_l0 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l1 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l2 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod1_l3 = inner_att_RA(d_model, pos=False, inner=29, outer=21, modalities=1)

        # self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)


        xeeg = self.inner_tf_mod0_l0(xeeg)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg = xeeg[:, :, 0].unsqueeze(dim=2)



        xeog = self.inner_tf_mod1_l0(xeog)
        xeog = self.inner_tf_mod1_l1(xeog)
        xeog = self.inner_tf_mod1_l2(xeog)

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog],dim=2)
        xeog = self.inner_tf_mod1_l3(xeog)
        xeog = xeog[:, :, 0].unsqueeze(dim=2)


        return xeeg, xeog

class EEG_SLEEP_MultiChannelMultiModal_merged_outermod_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_mod_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        x = torch.cat([xeeg, xeog], dim=3)

        x = self.tf(x)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiChannelMultiModal_bottleneck_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*3, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        def __init__(self, dmodel, pos, inner, outer, modalities, num_layers=1, heads=8, dim_feedforward = 1024):
            super().__init__()
            self.pos = pos
            if pos:
                self.pos_inner = PositionalEncoder(d_model=dmodel)

            enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
            self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        self.inner_tf_mod0_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.inner_tf_mod0_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        #
        # self.inner_tf_mod1_l0 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.inner_tf_mod1_l1 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.inner_tf_mod1_l2 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.inner_tf_mod1_l3 = inner_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.outer_tf_mod0_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        self.outer_tf_mod0_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        # self.outer_tf_mod1_l0 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.outer_tf_mod1_l1 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.outer_tf_mod1_l2 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)
        # self.outer_tf_mod1_l3 = outer_att(d_model, pos=False, inner=29, outer=21, modalities=1)

        self.common_bottleneck = nn.Parameter(torch.randn(1, 1, 5, 1, d_model))
        # self.common_bottleneck_outer = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod0 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))
        self.cls_token_mod1 = nn.Parameter(torch.randn(1, 1, 1, 1, d_model))

        self.inner_positional_embedding_0 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.inner_positional_embedding_1 = huy_pos_inner(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)
        self.outer_positional_embedding = huy_pos_outer(d_model, pos=False, inner=29, outer=21, modalities=1, channels=1)



    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        xeeg = self.inner_positional_embedding_0(xeeg)
        xeog = self.inner_positional_embedding_1(xeog)

        common =  self.common_bottleneck.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        # Inner layer 0
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:,:,:-5], xeeg[:,:,-5:]


        # xeog = torch.cat([xeog, common], dim=2)
        xeog = self.inner_tf_mod0_l0(xeog)
        # xeog, common = xeog[:,:,:-5], xeog[:,:,-5:]

        # Inner layer 1
        # xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        # xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod0_l1(xeog)
        # xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 2
        xeeg = torch.cat([xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l2(xeeg)
        xeeg, common = xeeg[:, :, :-5], xeeg[:, :, -5:]

        xeog = torch.cat([xeog, common],dim=2)
        xeog = self.inner_tf_mod0_l2(xeog)
        xeog, common = xeog[:, :, :-5], xeog[:, :, -5:]

        # Inner layer 3
        cls_token_eeg = self.cls_token_mod0.repeat( xeeg.shape[0], xeeg.shape[1], 1, xeeg.shape[3], 1)
        xeeg = torch.cat([cls_token_eeg, xeeg, common], dim=2)
        xeeg = self.inner_tf_mod0_l3(xeeg)
        xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, -5:]

        cls_token_eog = self.cls_token_mod1.repeat( xeog.shape[0], xeog.shape[1], 1, xeog.shape[3], 1)
        xeog = torch.cat([cls_token_eog, xeog, common],dim=2)
        xeog = self.inner_tf_mod0_l3(xeog)
        xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, -5:]


        common = common.mean(dim=2).unsqueeze(dim=2)

        # xeeg = self.outer_positional_embedding(xeeg)
        # xeog = self.outer_positional_embedding(xeog)
        # common = self.outer_positional_embedding(common)


        x = torch.cat([xeeg, common, xeog], dim=2)
        x = self.outer_tf_mod0_l0(x)
        x = self.outer_tf_mod0_l1(x)
        x = self.outer_tf_mod0_l2(x)
        x = self.outer_tf_mod0_l3(x)


        # Outer layer 0

        # xeeg = torch.cat([xeeg, common], dim=2)
        # xeeg = self.outer_tf_mod0_l0(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)
        #
        # xeog = torch.cat([xeog, common], dim=2)
        # xeog = self.outer_tf_mod1_l0(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)
        #
        # # Outer layer 1
        #
        # xeeg = torch.cat([xeeg, common], dim=2)
        # xeeg = self.outer_tf_mod0_l1(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)
        #
        # xeog = torch.cat([xeog, common], dim=2)
        # xeog = self.outer_tf_mod1_l1(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)
        #
        # # Outer layer 2
        #
        # xeeg = torch.cat([xeeg, common], dim=2)
        # xeeg = self.outer_tf_mod0_l2(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)
        #
        # xeog = torch.cat([xeog, common], dim=2)
        # xeog = self.outer_tf_mod1_l2(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)
        #
        # # Outer layer 3
        #
        # xeeg = torch.cat([xeeg, common], dim=2)
        # xeeg = self.outer_tf_mod0_l3(xeeg)
        # xeeg, common = xeeg[:, :, 0].unsqueeze(dim=2), xeeg[:, :, 1].unsqueeze(dim=2)
        #
        # xeog = torch.cat([xeog, common], dim=2)
        # xeog = self.outer_tf_mod1_l3(xeog)
        # xeog, common = xeog[:, :, 0].unsqueeze(dim=2), xeog[:, :, 1].unsqueeze(dim=2)
        #
        # x = torch.cat([xeeg, common, xeog], dim=2)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiChannelMultiModal_concat_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128*2#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        x = torch.cat([xeeg, xeog], dim=-1)

        x = self.tf(x)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiChannelMultiModal_concat_EEG_EOG_EMG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128*3#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        layers = [ "huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")
        xemg = einops.rearrange(xemg,"b outer mod f inner -> b outer inner mod f")

        x = torch.cat([xeeg, xeog, xemg], dim=-1)

        x = self.tf(x)

        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiChannelMultiModal_merged_with_diff_FC_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel=0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model = 128  # 64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        layers = ["huy_pos_inner", "inner_att_cls_aggr", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner=30, outer=21, modalities=2, heads=8,
                                        layers=layers, num_layers=4, pos=False)

    def forward(self, x, inits=None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        x = torch.cat([xeeg, xeog], dim=-2)

        x = self.tf(x)

        if len(x.shape) > 2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiChannelMultiModal_merged_innerouter_with_diff_FC_EEG_EOG(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel=0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model = 128  # 64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))

        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),

                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        layers = ["huy_pos_inner", "inner_mod_att_diff_FC_cls", "huy_pos_outer", "outer_mod_att_inner_diff_FC"]
        # print("Our layers are: \n {}".format(layers))
        self.tf = Multi_Transformer(d_model, inner=30, outer=21, modalities=2, heads=8,
                                        layers=layers, num_layers=4, pos=False)
        # layers = ["outer_att"]
        # self.outer_0_tf = Multi_Transformer(d_model, inner=30, outer=21, modalities=2, heads=8, layers=layers, num_layers=4, pos=False)
        # self.outer_1_tf = Multi_Transformer(d_model, inner=30, outer=21, modalities=2, heads=8, layers=layers, num_layers=4, pos=False)

    def forward(self, x, inits=None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

        xeeg = einops.rearrange(xeeg,"b outer mod f inner -> b outer inner mod f")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")

        x = torch.cat([xeeg, xeog], dim=-2)

        x = self.tf(x)
        # xeeg = self.outer_0_tf(x[:,:,:,0].unsqueeze(dim=3))
        # xeog = self.outer_1_tf(x[:,:,:,1].unsqueeze(dim=3))
        # x = torch.cat([xeeg, xeog], dim=-2)

        if len(x.shape) > 2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_MultiChannelMultiModal_late_late(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))

        self.fc_out_eeg = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Dropout(0.1),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        self.fc_out_eog = nn.Sequential(
            # nn.BatchNorm1d(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, fc_inner),
            nn.ReLU(),
            # nn.Dropout(0.45),
            nn.Linear(fc_inner, num_classes),
            nn.Softmax(dim=1)
        )

        self.fc_out_emg = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Dropout(0.1),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

        self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        layers = [ "huy_pos_inner", "inner_att","aggregation_att_contx_inner", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        self.tf_eeg = Multi_Transformer(d_model*2, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)

        self.tf_emg = Multi_Transformer(d_model, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)

        self.tf_eog = Multi_Transformer(d_model*2, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,1:,:] #mat
        xemg = x[2][:,:,:,1:,:] #mat
        xeog = x[1][:,:,:,1:,:] #mat

                # x = x[0][:,:,:,0,1:,:] #npz
        xeeg = einops.rearrange(xeeg,"b outer ch f inner -> b outer inner (ch f)").unsqueeze(dim=-2)
        xemg = einops.rearrange(xemg,"b outer ch f inner -> b outer inner (ch f)").unsqueeze(dim=-2)
        xeog = einops.rearrange(xeog,"b outer ch f inner -> b outer inner (ch f)").unsqueeze(dim=-2)

        xeeg = self.tf_eeg(xeeg)
        xeog = self.tf_eog(xeog)
        xemg = self.tf_emg(xemg)

        # x = torch.cat([xeeg, xeog, xemg], dim=-1)

        if len(xeeg.shape)>2:
            xeeg = xeeg.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        if len(xeog.shape)>2:
            xeog = xeog.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        if len(xemg.shape)>2:
            xemg = xemg.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)

        xeeg = self.fc_out_eeg(xeeg).unsqueeze(dim=-1)
        xeog = self.fc_out_eog(xeog).unsqueeze(dim=-1)
        xemg = self.fc_out_emg(xemg).unsqueeze(dim=-1)

        x = torch.cat([xeeg, xeog, xemg], dim=-1).mean(dim=-1).squeeze()

        return x

class EEG_SLEEP_Vilbert(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        # configs = BertConfig(128)
        # configs.num_hidden_layers = 5
        # configs.v_num_hidden_layers = 5
        # configs.v_feature_size = 128
        # configs.v_hidden_size = 128
        # configs.hidden_size = 128
        # configs.v_num_attention_heads = 8
        # configs.num_attention_heads = 8
        
        self.tf = MyViLBERT(128)
        d_model =  29*128*2#64*8
        fc_inner = 1024
        num_classes = 5
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Dropout(0.45),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

    def forward(self, x, inits= None):

        xeeg = x[0][:,:,:,0,1:,:]
        xeog = x[1][:,:,:,0,1:,:]
        xeeg = einops.rearrange(xeeg,"b outer f k inner -> (b outer) inner (f k) ")
        xeog = einops.rearrange(xeog,"b outer f k inner -> (b outer) inner (f k) ")
        x = self.tf(xeeg, xeog)
        x = torch.cat(x,dim=2)
        x = einops.rearrange(x,"b inner f -> b (inner f)")
        # if len(x.shape)>2:
        #     x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_EDF78_2(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  256#64*8
        fc_inner = 1024
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        nn.Dropout(0.45),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        dmodel = 128
        # layers = [ "fourier_pos", "inner_mod_att","inner_cross_att","aggregation_att_contx_inner","fourier_pos", "outer_cross_att"]
        layers_eeg = [ "fourier_pos", "inner_att"]
        layers_eog = [ "fourier_pos", "inner_att"]
        layers_common_1 = ["aggregation_att_contx_inner"]
        layers_common_2 = ["outer_att"]

        # print("Our layers are: \n {}".format(layers))
        self.tf_eeg = Multi_Transformer(dmodel, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers_eeg, num_layers=4, pos = False)

        self.tf_eog = Multi_Transformer(dmodel, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers_eog, num_layers=2, pos = False)
        self.tf_common = Multi_Transformer(dmodel, inner= 30, outer = 21, modalities=2, heads=8,
                                     layers = layers_common_1, num_layers=4, pos = False)

        # self.tf_eeg_2 = Multi_Transformer(dmodel, inner= 1, outer = 21, modalities=1, heads=8,
        #                              layers = layers_eeg_2, num_layers=4, pos = False)
        # self.tf_eog_2 = Multi_Transformer(dmodel, inner= 1, outer = 21, modalities=1, heads=8,
        #                              layers = layers_eog_2, num_layers=4, pos = False)

        self.tf_common_2 = Multi_Transformer(dmodel, inner= 30, outer = 21, modalities=2, heads=8,
                                     layers = layers_common_2, num_layers=4, pos = False)

        # self.tf = Multi_Transformer(dmodel, inner= 30, outer = 21, modalities=2, heads=8,
        #                              layers = layers, num_layers=1, pos = False)


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        # x = x[0]

        #Sleep_EDF
        xeeg = x[0][:,:,0,1:,:].unsqueeze(dim=2)
        xeeg = einops.rearrange(xeeg,"b  outer mod f inner -> b outer inner mod f")

        # x = einops.rearrange(x,"b outer mod inner k  -> b outer inner mod k ")
        xeog = x[1][:,:,0,1:,:].unsqueeze(dim=2)

        # xeeg = einops.rearrange(xeeg,"b outer f inner mod k  -> b outer inner (f mod) k ")
        # xeeg = einops.rearrange(xeeg,"b outer mod k inner -> b outer inner mod k ")
        xeog = einops.rearrange(xeog,"b outer mod f inner -> b outer inner mod f")
        # xeeg = self.tf1(xeeg)
        # xeog = self.tf2(xeog)

        # x = self.enc_0(xeeg,xeog)
        # xeog = self.enc_1(xeog)
        # x = torch.cat([xeeg, xeog], dim=3)
        # x = xeeg

        # x = xeeg
        # x = self.tf3(x)
        xeog = self.tf_eog(xeog)
        xeeg = self.tf_eeg(xeeg)
        x = torch.cat([xeeg, xeog], dim=3)
        x = self.tf_common(x)
        # xeeg = x[:,:,:,0,:].unsqueeze(dim=3)
        # xeog = x[:,:,:,1,:].unsqueeze(dim=3)
        # xeeg = self.tf_eeg_2(xeeg)
        # xeog = self.tf_eog_2(xeog)
        # x = torch.cat([xeeg, xeog], dim=3)
        x = self.tf_common_2(x)


        #Neonatal stft
        # x = x[:,0,:,:,:]
        # Neonatal signal
        # x = x.flatten(start_dim=0,end_dim=1)
        # x = x.permute(0,2,1,3)

        # x = x.mean(2)
        # x = self.enc_0(x)

        # if len(x.shape) > 2:
        #     x = x.flatten(start_dim=0, end_dim=1)
        # x, h = self.gru(x)
        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)

        return x

class EEG_SLEEP_EDF_Fusion(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  128#64*8
        fc_inner = 32
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        nn.BatchNorm1d(d_model),
                        nn.Dropout(0.45),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        # self.avg = nn.AvgPool2d(kernel_size=(1, 1500))


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        xeeg = x[0]
        xeog = x[1]
        if len(x_shape)>5:
            xeeg = xeeg.flatten(start_dim=0,end_dim=1)
            xeog = xeog.flatten(start_dim=0,end_dim=1)
        #Sleep EDF78
        # x = x[:,0,:,:,:]
        # x = x.permute(0,1,2,3)
        #Neonatal stft
        # x = x[:,0,:,:,:]
        # Neonatal signal
        xeeg = xeeg.flatten(start_dim=0,end_dim=1)
        xeog = xeog.flatten(start_dim=0,end_dim=1)
        # x = x.permute(0,2,1,3)
        # x_inner_shape = x.shape
        xeeg = self.enc_0(xeeg).flatten(start_dim=1,end_dim=2)
        xeog = self.enc_1(xeog).flatten(start_dim=1,end_dim=2)

        x = torch.cat([xeeg,xeog], dim=2)
        x_inner_shape = x.shape

        x = x.view([x_shape[0], x_shape[1], x_inner_shape[2],-1])
        x = self.enc_2(x)

        # if len(x.shape) > 2:
        #     x = x.flatten(start_dim=0, end_dim=1)
        # x, h = self.gru(x)
        if len(x.shape)>2:
            x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_EDF_CNN(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model =  240#64*8
        fc_inner = 32
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        nn.BatchNorm1d(d_model),
                        nn.Dropout(0.45),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        # self.avg = nn.AvgPool2d(kernel_size=(1, 1500))


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        x = x[0]

        x = self.enc_0(x)
        x_inner_shape = x.shape

        # x = x.view([x_shape[0], x_shape[1], x_inner_shape[3],-1]).mean(dim=2)
        # x = self.enc_1(x)

        # if len(x.shape) > 2:
        #     x = x.flatten(start_dim=0, end_dim=1)
        # x, h = self.gru(x)
        if len(x.shape)>2:
            x = x.flatten(start_dim=1)
        x = self.fc_out(x)
        return x

class EEG_SLEEP_EDF_Seq(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder"]

        d_model =  1024
        fc_inner = 32
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        self.gru = nn.LSTM(d_model, d_model, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        nn.BatchNorm1d(d_model*2),
                        nn.Dropout(0.45),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

    def forward(self, x):
        # z = x[0][:,:,:,1:2]
        # x = x[0]
        x = self.enc_0(x)
        # x = x.flatten(start_dim=0, end_dim=1)

        x, _ = self.gru(x)
        if len(x.shape) > 2:
            x = x.flatten(start_dim=0, end_dim=1)
        x = self.fc_out(x)
        return x

class EEG_Global_LSTM(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder"]

        d_model =  128
        fc_inner = 32
        num_classes = 5
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        self.seq = nn.GRU(1028, d_model, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.hidden = None
        self.fc_out = nn.Sequential(
                        nn.BatchNorm1d(d_model*2),
                        nn.Dropout(0.45),
                        nn.Linear(d_model*2, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

    def forward(self, x, inits):

        # z = x[0][:,:,:,1:2]
        # x = x[0]
        x = self.enc_0(x)
        # x = x.flatten(start_dim=0, end_dim=1)

        if inits.sum() == 0:
            x, h = self.seq(x.unsqueeze(dim=1), self.hidden)
        else:
            m = torch.argmax(inits)
            x2, h = self.seq(x[m:].unsqueeze(dim=1))
            if m>0:
                x1 , _ = self.seq(x[:m].unsqueeze(dim=1),  self.hidden)
                x = torch.cat([x1, x2], dim=0)
            else:
                x = x2
        self.hidden = copy.deepcopy((h[0].detach(),h[1].detach()))
        if len(x.shape) > 2:
            x = x.flatten(start_dim=0, end_dim=1)
        x = self.fc_out(x)
        return x

class EEG_CNN_Ch_T(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder"]

        d_model = 1696
        fc_inner = 32
        num_classes = 2
        self.channel = channel
        print("We are processing channel {}".format(self.channel))

        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))

        # self.encs = lambda x: [getattr(self,"enc_{}".format(i))(x[:,:,i,:]).flatten(start_dim=1) for i in range(x.shape[2])]

        # self.pos_emb = PositionalEncoder(d_model=128)
        # enc_0 = nn.TransformerEncoderLayer(d_model=128 , nhead=8)
        # self.self_attention = nn.TransformerEncoder(encoder_layer=enc_0, num_layers=2)
        # self.small_lstm = LSTM(128, hidden_size = 128, num_layers= 1, bidirectional=False, merge_func = lambda x:x.flatten(start_dim=1))

        # #
        # self.pos_emb_1 = PositionalEncoder(d_model=d_model)
        # enc_1 = nn.TransformerEncoderLayer(d_model=d_model , nhead=8)
        # self.self_attention_1 = nn.TransformerEncoder(encoder_layer=enc_1, num_layers=4)

        # self.pos_emb_2 = PositionalEncoder(d_model=d_model)
        # enc_2 = nn.TransformerEncoderLayer(d_model=d_model , nhead=8)
        # self.self_attention_2 = nn.TransformerEncoder(encoder_layer=enc_2, num_layers=4)

        # self.big_lstm = LSTM(128, hidden_size = 64, num_layers= 2, bidirectional=True, merge_func = lambda x:x.flatten(start_dim=2))
        # self.convX2 = nn.Sequential(
        #     nn.ZeroPad2d((2, 2, 1, 1)),
        #     nn.Conv2d(64*encoder_filters, 128*encoder_filters, kernel_size=(2, 5), stride=(1, 1)),
        #     nn.ReLU( ),
        # )
        # self.max_pool = nn.MaxPool2d(kernel_size=(1, 2))
        #
        # self.convX3 = nn.Sequential(
        #     nn.ZeroPad2d((2, 2, 0, 0)),
        #     nn.Conv2d(64*encoder_filters, 16*encoder_filters, kernel_size=(1, 5), stride=(1, 1)),
        #     nn.ReLU()
        # )
        # self.avg = nn.AvgPool2d(kernel_size=(1,56*4))

        # self.pos_emb = PositionalEncoder(d_model=128)
        # self.pos_emb_1 = PositionalEncoder(d_model=128)

        # enc = nn.TransformerEncoderLayer(d_model=1696 , nhead=8)
        # self.self_attention = nn.TransformerEncoder(encoder_layer=enc, num_layers=1)
        #

        # dec_0 = nn.TransformerDecoderLayer(d_model=128, nhead=16)
        # self.dec_att_01 = nn.TransformerDecoder(decoder_layer=dec_0, num_layers=1)
        #
        # dec_0 = nn.TransformerDecoderLayer(d_model=128, nhead=16)
        # self.dec_att_10 = nn.TransformerDecoder(decoder_layer=dec_0, num_layers=1)


        self.fc_out = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Dropout(0.45),
            nn.Linear(d_model, fc_inner),
            nn.ReLU(),
            # nn.Dropout(0.45),
            nn.Linear(fc_inner, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # z = x[0][:,:,:,1:2]
        x = x[0]#[:,:,:,0:8]
        x_shape = x.shape
        x = x.flatten(start_dim=0, end_dim=1).unsqueeze(dim=1)
        # z = z.flatten(start_dim=0, end_dim=2)

        x = self.enc_0(x)
        # m = []
        # for i in range(8):
        #     for j in range(i+1,8):
        #         m.append(x[:,:,i].unsqueeze(dim=2))
        #         m.append(x[:,:,j].unsqueeze(dim=2))
        # m = torch.cat(m,dim=2)
        # x = self.convX2(x)
        #
        # x = self.max_pool(x)
        # x = self.convX3(x)


        # x =  self.avg(x)
        x = x.permute(0,2,1,3).flatten(start_dim=2)
        torch.manual_seed(1)
        mask_token = torch.randn(x_shape[0], 1,1696).cuda()
        x = torch.cat([x , mask_token],dim=1)
        # x0 = x[:,0,:].view([x_shape[0],x_shape[1],-1])
        # x1 = x[:,1,:].view([x_shape[0],x_shape[1],-1])
        # x3 = self.pos_emb(x[:,2,:].view([x_shape[0],x_shape[1],-1]))
        # x11 = self.dec_att_01(x0,x1)
        # x00 = self.dec_att_10(x1,x0)
        # x = torch.cat([x00,x11],dim=2)
        # xi = []
        # for i in range(8):
        #     xi.append( self.pos_emb(x[:,i,:].view([x_shape[0],x_shape[1],-1])))
        # x = []
        # for i in range(8):
        #     m = []
        #     for j in range(8):
        #         m.append(self.dec_att_01(xi[i], xi[j]))
        #     x.append(torch.cat(m,dim=2))
        # x = torch.cat(x,dim=2)
        x = self.self_attention(x)
        # print(x.shape)
        # x = x.view([x_shape[0],-1])
        # x = x.flatten(start_dim=1)

        # z = self.enc_0(z)
        # z= x[:,0]
        # x= x[:,1]
        # x = x.view([x_shape[0],x_shape[1],-1])
        # z = z.view([x_shape[0],x_shape[1],-1])
        # x = x.view([x_shape[0]*x_shape[1],x_shape[3],-1])
        # print(x.shape)
        # x = x.view([x_shape[0]*x_shape[1],x_shape[2],-1])
        # x = self.pos_emb(x)
        # x = self.self_attention(x)
        # x = x.flatten(start_dim=1)

        # x = self.small_lstm(x)


        # x = self.pos_emb_1(x)
        # x = self.self_attention_1(x)
        # z = self.pos_emb_2(z)
        # z = self.self_attention_2(z)
        # x = self.big_lstm(x)
        x = self.fc_out(x[:,-1])
        # z = self.fc_out(z.flatten(start_dim=0, end_dim=1)).unsqueeze(dim=1)
        # x = torch.cat([x,z],dim=1).mean(dim=1)
        return x

class EEG_Minirocket(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        super().__init__()
        d_model = 9996
        fc_inner = 32
        num_classes = 2

        self.fc_out = nn.Sequential(
            nn.BatchNorm1d(d_model),
            # nn.Dropout(0.45),
            # nn.Linear(d_model, fc_inner),
            # nn.ReLU(),
            # nn.Dropout(0.45),
            nn.Linear(d_model, num_classes),
            nn.Softmax(dim=1)
        )
        self.first_passed = False
        import numpy as np
        self.par1 = np.ndarray([0])
        self.par2 = np.ndarray([0])
        self.par3 = np.ndarray([0])
    def forward(self, x):
        x = x[0][:,:,0,:].squeeze()

        if (not self.first_passed):
            (self.par1, self.par2, self.par3) = fit(x.detach().cpu().numpy())
            self.first_passed = True
        X_training_transform = transform(x.detach().cpu().numpy(), (self.par1, self.par2, self.par3))
        X_training_transform = torch.FloatTensor(X_training_transform).cuda()
        out = self.fc_out(X_training_transform)

        return out

class EEG_Fairseq(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        super().__init__()
        encs_base = ["EEG_Encoder"]

        d_model = 576
        fc_inner = 32
        num_classes = 2
        self.channel = channel
        print("We are processing channel {}".format(self.channel))

        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))

        class args:
            def __init__(self):
                self.encoder_glu = True
                self.encoder_conv_type= "lightweight"
                self.dropout= 0.1
                self.relu_dropout= 0.1
                self.input_dropout= 0.1
                self.encoder_embed_dim= 576
                self.encoder_conv_dim= 128
                self.encoder_ffn_embed_dim= 1024
                self.weight_softmax = False
                self.encoder_attention_heads = 8
                self.weight_dropout = 0.0
                self.encoder_normalize_before = True
        arg = args()
        # args = {"encoder_glu": True, "encoder_conv_type": "lightweight", "dropout": 0.1, "relu_dropout": 0.1,
        #         "input_dropout": 0.1, "encoder_embed_dim": 512, "encoder_conv_dim": 128}

        self.seq = LightConvEncoderLayer(arg, 5)
        self.pos_emb = PositionalEncoder(d_model=d_model)

        self.fc_out = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Dropout(0.45),
            nn.Linear(d_model, fc_inner),
            nn.ReLU(),
            # nn.Dropout(0.45),
            nn.Linear(fc_inner, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x[0]

        x_shape = x.shape

        x = x.view([x_shape[0]*x.shape[1], x_shape[2], x_shape[3], x_shape[4]])

        x = self.enc_0(x)
        x = x.view([x_shape[0], x_shape[1], -1])

        x = self.seq(self.pos_emb(x), None)
        x = x.view([x_shape[0] * x_shape[1], -1])
        x = self.fc_out(x)

        return x

class EEG_StandAlone_Att(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        super().__init__()
        self.stem = nn.Sequential(AttentionStem(1,64,kernel_size=[1,5],stride=1, padding=[2,2,0,0]), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(1,2))
        self.stem1 = nn.Sequential(AttentionStem(64,128,kernel_size=[2,5],stride=1, padding=[2,2,1,1]), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(1,2))
        self.stem2 = nn.Sequential(AttentionStem(128,16,kernel_size=[2,5],stride = 1, padding=[2,2,0,0]), nn.BatchNorm2d(64), nn.ReLU(), nn.AvgPool2d((1,56)))
        # self.b1 = Bottleneck(in_channels=64,out_channels=16, stride=2)
        # self.avg = nn.AvgPool2d((1,56))

        d_model = 128
        fc_inner = 32
        num_classes = 2

        self.fc_out = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Dropout(0.45),
            nn.Linear(d_model, fc_inner),
            nn.ReLU(),
            # nn.Dropout(0.45),
            nn.Linear(fc_inner, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x[0]
        x = self.stem(x)
        x = self.stem1(x)
        x = self.stem2(x)
        # x = self.b1(x)
        # x =

class EEG_CNN_Late_Prob_Fusion(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder"]

        d_model = 128
        fc_inner = 32
        num_classes = 2
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))

        self.fc_out_0 = nn.Sequential(
                        nn.BatchNorm1d(d_model),
                        nn.Dropout(0.45),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.fc_out_1 = nn.Sequential(
                        nn.BatchNorm1d(d_model),
                        nn.Dropout(0.45),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.fc_out_2 = nn.Sequential(
                        nn.BatchNorm1d(d_model),
                        nn.Dropout(0.45),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.fc_out_3 = nn.Sequential(
                        nn.BatchNorm1d(d_model),
                        nn.Dropout(0.45),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.fc_out_4 = nn.Sequential(
                        nn.BatchNorm1d(d_model),
                        nn.Dropout(0.45),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.fc_out_5 = nn.Sequential(
                        nn.BatchNorm1d(d_model),
                        nn.Dropout(0.45),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.fc_out_6 = nn.Sequential(
                        nn.BatchNorm1d(d_model),
                        nn.Dropout(0.45),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.fc_out_7 = nn.Sequential(
                        nn.BatchNorm1d(d_model),
                        nn.Dropout(0.45),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )

    def forward(self, x):
        # z = x[0][:,:,:,1:2]
        x = x[0]#[:,:,:,0:8]
        x = self.enc_0(x).flatten(start_dim=2)
        m = []
        m.append(self.fc_out_0(x[:, 0, :]).unsqueeze(dim=1))
        m.append(self.fc_out_1(x[:, 1, :]).unsqueeze(dim=1))
        m.append(self.fc_out_2(x[:, 2, :]).unsqueeze(dim=1))
        m.append(self.fc_out_3(x[:, 3, :]).unsqueeze(dim=1))
        m.append(self.fc_out_4(x[:, 4, :]).unsqueeze(dim=1))
        m.append(self.fc_out_5(x[:, 5, :]).unsqueeze(dim=1))
        m.append(self.fc_out_6(x[:, 6, :]).unsqueeze(dim=1))
        m.append(self.fc_out_7(x[:, 7, :]).unsqueeze(dim=1))

        x = torch.cat(m,dim=1).mean(dim=1)
        return x


class Epilepsy_Base_Model(nn.Module):
    def __init__(self, encs=[None], encoder_filters=2, channel = 0):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        encs_base = ["EEG_Encoder", "EEG_Embedding"]

        d_model = 336#64*8
        fc_inner = 1024
        num_classes = 2
        for i, enc in enumerate(encs):
            if enc != None:
                setattr(self, "enc_{}".format(i), enc)
            else:
                setattr(self, "enc_{}".format(i), globals()[encs_base[i]](encoder_filters))
        # self.output_seq2seq = seq2seq_GRU(d_model, num_classes)
        n_layers = 1
        # self.gru = nn.GRU(d_model, d_model, n_layers,
        #                   dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc_out = nn.Sequential(
                        # nn.BatchNorm1d(d_model),
                        nn.Dropout(0.1),
                        nn.Linear(d_model, fc_inner),
                        nn.ReLU(),
                        # nn.Dropout(0.45),
                        nn.Linear(fc_inner, num_classes),
                        nn.Softmax(dim=1)
                    )
        self.avg = nn.AvgPool2d(kernel_size=(49, 1))
        # dmodel = 128
        # layers = [ "huy_pos_inner", "inner_att","aggregation_att_contx_inner", "huy_pos_outer", "outer_att"]
        # print("Our layers are: \n {}".format(layers))
        # self.tf = Multi_Transformer(dmodel, inner= 30, outer = 21, modalities=1, heads=8,
        #                              layers = layers, num_layers=4, pos = False)


    def forward(self, x, inits= None):
        x_shape =x[0].shape

        x = x[0]
        x = einops.rearrange(x,"b outer f k inner -> (b outer) f k inner")
        x = self.enc_0(x)
        x = einops.rearrange(x,"a f k inner -> a (f k inner)")

        x = self.fc_out(x)
        return x

