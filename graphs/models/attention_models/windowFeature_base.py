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
from vilbert.vilbert import BertForMultiModalPreTraining, BertConfig
from graphs.models.attention_models.ViLBERT import MyViLBERT

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
        print("Our layers are: \n {}".format(layers))
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
        x = x[0][:,:,:,0,1:,:]
        x = einops.rearrange(x,"b outer f k inner -> b outer inner f k ")

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
        layers_common_1 = ["inner_cross_att","aggregation_att_contx_inner"]
        layers_eeg_2 = [ "fourier_pos", "outer_att"]
        layers_eog_2 = [ "fourier_pos", "outer_att"]
        layers_common_2 = ["outer_cross_att"]

        # print("Our layers are: \n {}".format(layers))
        self.tf_eeg = Multi_Transformer(dmodel, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers_eeg, num_layers=4, pos = False)

        self.tf_eog = Multi_Transformer(dmodel, inner= 30, outer = 21, modalities=1, heads=8,
                                     layers = layers_eog, num_layers=4, pos = False)
        self.tf_common = Multi_Transformer(dmodel, inner= 30, outer = 21, modalities=2, heads=8,
                                     layers = layers_common_1, num_layers=2, pos = False)

        self.tf_eeg_2 = Multi_Transformer(dmodel, inner= 1, outer = 21, modalities=1, heads=8,
                                     layers = layers_eeg_2, num_layers=4, pos = False)
        self.tf_eog_2 = Multi_Transformer(dmodel, inner= 1, outer = 21, modalities=1, heads=8,
                                     layers = layers_eog_2, num_layers=4, pos = False)

        self.tf_common_2 = Multi_Transformer(dmodel, inner= 30, outer = 21, modalities=2, heads=8,
                                     layers = layers_common_2, num_layers=2, pos = False)

        # self.tf = Multi_Transformer(dmodel, inner= 30, outer = 21, modalities=2, heads=8,
        #                              layers = layers, num_layers=1, pos = False)


    def forward(self, x, inits= None):
        x_shape =x[0].shape
        # x = x[0]

        #Sleep_EDF
        xeeg = x[0][:,:,:,0,1:,:]
        xeeg = einops.rearrange(xeeg,"b  outer mod k inner -> b outer inner mod k ")

        # x = einops.rearrange(x,"b outer mod inner k  -> b outer inner mod k ")
        xeog = x[1][:,:,:,0,1:,:]

        # xeeg = einops.rearrange(xeeg,"b outer f inner mod k  -> b outer inner (f mod) k ")
        # xeeg = einops.rearrange(xeeg,"b outer mod k inner -> b outer inner mod k ")
        xeog = einops.rearrange(xeog,"b outer mod k inner -> b outer inner mod k ")
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
        xeeg = x[:,:,:,0,:].unsqueeze(dim=3)
        xeog = x[:,:,:,1,:].unsqueeze(dim=3)
        xeeg = self.tf_eeg_2(xeeg)
        xeog = self.tf_eog_2(xeog)
        x = torch.cat([xeeg, xeog], dim=3)
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
