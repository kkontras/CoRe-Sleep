
""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
import math
from torch.autograd import Variable
# from graphs.models.custom_layers.attention import Attention
from graphs.models.custom_layers.coattention import CoattentionNet
from graphs.models.attention_models.utils.positionalEncoders import PositionalEncoder
from graphs.models.custom_layers.eeg_encoders import *

class EEG_ATT_1(nn.Module):
    def __init__(self, encs=[None]):
        super().__init__()
        if encs[0]:
            self.encoder = encs[0]
        else:
            self.encoder = EEG_Encoder(2)
        self.pos = PositionalEncoder(d_model= 900)
        encoder_layer = nn.TransformerEncoderLayer(d_model=900, nhead=9)
        self.att = nn.TransformerEncoder(encoder_layer, num_layers= 6)
        self.fc1 = nn.Linear(900, 32)
        self.fc2 = nn.Linear(32, 2)
        self.dropout = torch.nn.Dropout(p=0.45)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(900)

    def forward(self, x):
        x = x[0][:,:,0,0,:]
        # x_f,_,_,_ = self.encoder(x[0].view([x[0].shape[0]*x[0].shape[1],x[0].shape[2],x[0].shape[3],x[0].shape[4]]))
        xx = self.att(self.pos(x.view([x.shape[0],x.shape[1],900])))
        xx = self.dropout(self.dense1_bn(xx.view([x.shape[0]*x.shape[1],900])))
        xx = self.dropout(self.relu(self.fc1(xx)))
        xx = F.softmax(self.fc2(xx), dim=1)
        return xx

class EEG_CNN_6(nn.Module):
    def __init__(self, encs=[None]):
        super().__init__()
        if encs[0]:
            self.encoder = encs[0]
        else:
            self.encoder = EEG_Encoder(1)
        self.fc1 = nn.Linear(20 * 75, 10)
        self.fc2 = nn.Linear(10, 2)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(20 * 75)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x[0])
        xx = self.dropout(self.dense1_bn(torch.flatten(x[0], start_dim=1)))
        xx = self.dropout(self.sigmoid(self.fc1(xx)))
        xx = F.softmax(self.fc2(xx), dim=1)
        return xx


class EEG_CNN_7(nn.Module):
    def __init__(self, encs=[None]):
        super().__init__()
        if encs[0]:
            self.encoder = encs[0]
        else:
            self.encoder = EEG_Encoder(2)
        self.fc1 = nn.Linear(40 * 75, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(p=0.45)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.dense1_bn = nn.BatchNorm1d(40 * 75)

    def forward(self, x):
        x = self.encoder(x[0])
        xx = self.dropout(self.dense1_bn(torch.flatten(x, start_dim=1)))
        xx = self.dropout(self.relu(self.fc1(xx)))
        xx = self.softmax(self.fc2(xx))
        return xx

class EEG_CNN_7_E(nn.Module):
    def __init__(self, encs=[None]):
        super().__init__()
        if encs[0]:
            self.encoder = encs[0]
        else:
            self.encoder = EEG_Encoder_E_3(2)
        self.fc1 = nn.Linear(40 * 100, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(p=0.45)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(40 * 100)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x[0])
        xx = self.dropout(self.dense1_bn(torch.flatten(x, start_dim=1)))
        xx = self.dropout(self.relu(self.fc1(xx)))
        xx = self.softmax(self.fc2(xx))
        return xx


class EEG_CNN_8(nn.Module):
    def __init__(self, encs=[None]):
        super().__init__()
        if encs[0]:
            self.encoder = encs[0]
        else:
            self.encoder = EEG_Encoder(3)
        self.fc1 = nn.Linear(60 * 75, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(60 * 75)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x[0])
        xx = self.dropout(self.dense1_bn(torch.flatten(x[0], start_dim=1)))
        xx = self.dropout(self.sigmoid(self.fc1(xx)))
        xx = F.softmax(self.fc2(xx), dim=1)
        return xx


class EEG_CNN_9(nn.Module):
    def __init__(self, encs=[None]):
        super().__init__()
        if encs[0]:
            self.encoder = encs[0]
        else:
            self.encoder = EEG_Encoder(4)
        self.fc1 = nn.Linear(80 * 75, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(80 * 75)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x[0])
        xx = self.dropout(self.dense1_bn(torch.flatten(x[0], start_dim=1)))
        xx = self.dropout(self.sigmoid(self.fc1(xx)))
        xx = F.softmax(self.fc2(xx), dim=1)
        return xx


class STFT_CNN(nn.Module):
    def __init__(self, encs=[None]):
        super().__init__()
        if encs[0]:
            self.encoder = encs[0]
        else:
            self.encoder = STFT_Encoder(1)
        self.fc1 = nn.Linear(20 * 25, 10)
        self.fc2 = nn.Linear(10, 2)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(20 * 25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x[0])
        xx = self.dropout(self.dense1_bn(torch.flatten(x[0], start_dim=1)))
        xx = self.dropout(self.sigmoid(self.fc1(xx)))
        xx = F.softmax(self.fc2(xx), dim=1)
        return xx


class STFT_CNN_2(nn.Module):
    def __init__(self, encs=[None]):
        super().__init__()
        if encs[0]:
            self.encoder = encs[0]
        else:
            self.encoder = STFT_Encoder_Ε(2)
        self.fc1 = nn.Linear(40 * 25, 32)
        self.fc2 = nn.Linear(32, 2)
        self.dropout = torch.nn.Dropout(p=0.45)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(40 * 25)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.encoder(x[0])
        xx = self.dropout(self.dense1_bn(torch.flatten(x[0], start_dim=1)))
        xx = self.dropout(self.sigmoid(self.fc1(xx)))
        xx = self.softmax(self.fc2(xx))
        return xx

class STFT_CNN_E_2(nn.Module):
    def __init__(self, encs=[None]):
        super().__init__()
        if encs[0]:
            self.encoder = encs[0]
        else:
            self.encoder = STFT_Encoder_Ε(2)
        self.fc1 = nn.Linear(40 * 100, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(p=0.45)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(40 * 100)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.encoder(x[0])
        xx = self.dropout(self.dense1_bn(torch.flatten(x, start_dim=1)))
        xx = self.dropout(self.sigmoid(self.fc1(xx)))
        xx = self.softmax(self.fc2(xx))
        return xx



class STFT_CNN_3(nn.Module):
    def __init__(self, encs=[None]):
        super().__init__()
        if encs[0]:
            self.encoder = encs[0]
        else:
            self.encoder = STFT_Encoder(3)
        self.fc1 = nn.Linear(60 * 25, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = torch.nn.Dropout(p=0.45)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(60 * 25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x[0])
        xx = self.dropout(self.dense1_bn(torch.flatten(x[0], start_dim=1)))
        xx = self.dropout(self.sigmoid(self.fc1(xx)))
        xx = F.softmax(self.fc2(xx), dim=1)
        return xx


class STFT_CNN_4(nn.Module):
    def __init__(self, encs=[None]):
        super().__init__()
        if encs[0]:
            self.encoder = encs[0]
        else:
            self.encoder = STFT_Encoder(4)
        self.fc1 = nn.Linear(80 * 25, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(p=0.45)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(80 * 25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x[0])
        xx = self.dropout(self.dense1_bn(torch.flatten(x[0], start_dim=1)))
        xx = self.dropout(self.sigmoid(self.fc1(xx)))
        xx = F.softmax(self.fc2(xx), dim=1)
        return xx

class STFT_EEG_ATT(nn.Module):

    def __init__(self, encs=[None, None]):
        super().__init__()
        if encs[0]:
            self.encoder_eeg = encs[0]
        else:
            self.encoder_eeg = EEG_Encoder(2)
        if encs[1]:
            self.encoder_stft = encs[1]
        else:
            self.encoder_stft = STFT_Encoder(2)

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 2)
        self.dropout = torch.nn.Dropout(p=0.35)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(128)
        self.dense2_bn = nn.BatchNorm1d(32)

        # self.emb_fc = nn.Linear(1024,256)
        self.emb_fc_1 = nn.Linear(3000,128)
        self.emb_fc_2 = nn.Linear(1000,128)
        self.alpha = torch.nn.Parameter(torch.empty(128))
        self.b = torch.nn.Parameter(torch.empty(1))

    def forward(self, x):
        x_eeg = self.encoder_eeg(x[0])
        x_stft = self.encoder_stft(x[1])

        x_eeg = self.emb_fc_1(x_eeg[0].flatten(start_dim=1))
        x_stft = self.emb_fc_2(x_stft[0].flatten(start_dim=1))

        x_eeg_alpha = F.softmax(self.alpha * x_eeg + self.b,dim=1)
        x_stft_alpha = F.softmax(self.alpha * x_stft + self.b,dim=1)

        xcat = x_eeg_alpha*x_eeg + x_stft_alpha * x_stft

        xx = self.dropout(self.dense1_bn(xcat))
        xx = self.dropout(self.dense2_bn(self.relu(self.fc1(xx))))
        xx = F.softmax(self.fc2(xx), dim=1)
        return xx

class STFT_EEG_ATT_2(nn.Module):

    def __init__(self, encs=[None, None]):
        super().__init__()
        if encs[0]:
            self.encoder_eeg = encs[0]
        else:
            self.encoder_eeg = EEG_Encoder(2)
        if encs[1]:
            self.encoder_stft = encs[1]
        else:
            self.encoder_stft = STFT_Encoder(2)

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 2)
        self.dropout = torch.nn.Dropout(p=0.35)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(128)
        self.dense2_bn = nn.BatchNorm1d(32)

        # self.emb_fc = nn.Linear(1024,256)
        self.emb_fc_1 = nn.Linear(3000,128)
        self.emb_fc_2 = nn.Linear(1000,128)
        self.alpha = torch.nn.Parameter(torch.empty(128))
        self.b = torch.nn.Parameter(torch.empty(1))

        self.pos = PositionalEncoder(d_model= 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.att = nn.TransformerEncoder(encoder_layer, num_layers= 1)

    def forward(self, x):
        x_eeg = self.encoder_eeg(x[0])
        x_stft = self.encoder_stft(x[1])

        x_eeg = self.emb_fc_1(x_eeg[0].flatten(start_dim=1))
        x_stft = self.emb_fc_2(x_stft[0].flatten(start_dim=1))

        xcat_1 = torch.cat([x_eeg.unsqueeze(dim=1),x_stft.unsqueeze(dim=1)],dim = 1)
        xcat_1 = self.att(self.pos(xcat_1))

        x_eeg, x_stft = xcat_1[:,0], xcat_1[:,1]

        x_eeg_alpha = F.softmax(self.alpha * x_eeg + self.b,dim=1)
        x_stft_alpha = F.softmax(self.alpha * x_stft + self.b,dim=1)

        xcat = x_eeg_alpha*x_eeg + x_stft_alpha * x_stft

        xx = self.dropout(self.dense1_bn(xcat))
        xx = self.dropout(self.dense2_bn(self.relu(self.fc1(xx))))
        xx = F.softmax(self.fc2(xx), dim=1)
        return xx

class STFT_EEG_COATT(nn.Module):

    def __init__(self, encs=[None, None]):
        super().__init__()
        if encs[0]:
            self.encoder_eeg = encs[0]
        else:
            self.encoder_eeg = EEG_Encoder(2)
        if encs[1]:
            self.encoder_stft = encs[1]
        else:
            self.encoder_stft = STFT_Encoder(2)
        self.fc1 = nn.Linear(40 * 25 + 40 * 75, 32)
        self.fc2 = nn.Linear(32, 2)
        self.dropout = torch.nn.Dropout(p=0.45)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(40 * 25 + 40 * 75)
        self.sigmoid = nn.Sigmoid()
        self.att = CoattentionNet(embed_dim = 40)
        self.que_mlp = nn.Sequential(
            nn.Linear(40,10),
            nn.Tanh(),
        )

        self.img_mlp = nn.Sequential(
            nn.Linear(40, 10),
            nn.Tanh(),
        )

        self.dropout = nn.Dropout(0.2)

        self.final_mlp = nn.Linear(10, 2)

    def forward(self, x):
        x_eeg = self.encoder_eeg(x[0])
        x_stft = self.encoder_stft(x[1])
        q, v = self.att(x_eeg[0].squeeze(),x_stft[0].view([x_stft[0].shape[0],40,25]).permute(0,2,1))
        x = torch.cat([torch.flatten(q, start_dim=1), torch.flatten(v, start_dim=1)], dim=1)
        x = self.dropout(self.dense1_bn(x))
        x = self.dropout(self.sigmoid(self.fc1(x)))
        x = F.softmax(self.fc2(x), dim=1)
        return x

class STFT_EEG_CNN_1(nn.Module):
    def __init__(self, encs=[None, None]):
        super().__init__()
        if encs[0]:
            self.encoder_eeg = encs[0]
        else:
            self.encoder_eeg = EEG_Encoder(1)
        if encs[1]:
            self.encoder_stft = encs[1]
        else:
            self.encoder_stft = STFT_Encoder(1)

        self.fc1 = nn.Linear(20 * 25 + 20 * 75, 10)
        self.fc2 = nn.Linear(10, 2)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(20 * 25 + 20 * 75)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_eeg = self.encoder_eeg(x[0].squeeze(dim=1))
        x_stft = self.encoder_stft(x[1].squeeze(dim=1))
        xx = self.dropout(self.dense1_bn(
            torch.flatten(torch.cat([x_eeg, torch.flatten(x_stft, start_dim=3)], dim=3), start_dim=1)))
        xx = self.dropout(self.sigmoid(self.fc1(xx)))
        xx = F.softmax(self.fc2(xx), dim=1)
        return xx


class STFT_EEG_CNN_2(nn.Module):
    def __init__(self, encs=[None, None]):
        super().__init__()
        if encs[0]:
            self.encoder_eeg = encs[0]
        else:
            self.encoder_eeg = EEG_Encoder(2)
        if encs[1]:
            self.encoder_stft = encs[1]
        else:
            self.encoder_stft = STFT_Encoder(2)
        self.fc1 = nn.Linear(40 * 25 + 40 * 75, 32)
        self.fc2 = nn.Linear(32, 2)
        self.dropout = torch.nn.Dropout(p=0.45)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(40 * 25 + 40 * 75)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_eeg = self.encoder_eeg(x[0])
        x_stft = self.encoder_stft(x[1])
        xx = self.dropout(self.dense1_bn(
            torch.flatten(torch.cat([x_eeg, torch.flatten(x_stft, start_dim=3)], dim=3), start_dim=1)))
        xx = self.dropout(self.sigmoid(self.fc1(xx)))
        xx = F.softmax(self.fc2(xx), dim=1)
        return xx


class STFT_EEG_CNN_3(nn.Module):
    def __init__(self, encs=[None, None]):
        super().__init__()
        if encs[0]:
            self.encoder_eeg = encs[0]
        else:
            self.encoder_eeg = EEG_Encoder(3)
        if encs[1]:
            self.encoder_stft = encs[1]
        else:
            self.encoder_stft = STFT_Encoder(3)
        self.fc1 = nn.Linear(60 * 25 + 60 * 75, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = torch.nn.Dropout(p=0.45)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(60 * 25 + 60 * 75)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_eeg = self.encoder_eeg(x[0])
        x_stft = self.encoder_stft(x[1])
        xx = self.dropout(self.dense1_bn(
            torch.flatten(torch.cat([x_eeg, torch.flatten(x_stft, start_dim=3)], dim=3), start_dim=1)))
        xx = self.dropout(self.sigmoid(self.fc1(xx)))
        xx = F.softmax(self.fc2(xx), dim=1)
        return xx


class STFT_EEG_CNN_4(nn.Module):
    def __init__(self, encs=[None, None]):
        super().__init__()
        if encs[0]:
            self.encoder_eeg = encs[0]
        else:
            self.encoder_eeg = EEG_Encoder(4)
        if encs[1]:
            self.encoder_stft = encs[1]
        else:
            self.encoder_stft = STFT_Encoder(4)
        self.fc1 = nn.Linear(80 * 25 + 80 * 75, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(p=0.45)
        self.relu = torch.nn.ReLU()
        self.dense1_bn = nn.BatchNorm1d(80 * 25 + 80 * 75)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_eeg = self.encoder_eeg(x[0])
        x_stft = self.encoder_stft(x[1])
        xx = self.dropout(self.dense1_bn(
            torch.flatten(torch.cat([x_eeg, torch.flatten(x_stft, start_dim=3)], dim=3), start_dim=1)))
        xx = self.dropout(self.sigmoid(self.fc1(xx)))
        xx = F.softmax(self.fc2(xx), dim=1)
        return xx


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(768, 512 // factor, bilinear)
        self.up2 = Up(768, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.fc1 = nn.Linear(512*187, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xx = self.fc1(torch.flatten(x4, start_dim=1))
        xx = self.dropout(xx)
        xx = F.log_softmax(self.fc2(xx),dim=1)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits, xx

class EEG_CNN_Amir (nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 =  nn.Conv2d(1,10,kernel_size=(2,10),padding=(1,5),stride=(1,1))
        self.conv2 =  nn.Conv2d(10,20,kernel_size=(1,5),padding=(0,2),stride=(1,1))
        self.conv3 =  nn.Conv2d(20,20,kernel_size=(4,1),padding=(0,0),stride=(1,1))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool_time = nn.MaxPool2d(kernel_size=(1, 6))
        self.fc1 = nn.Linear(20*75, 10)
        self.fc2 = nn.Linear(10, 2)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.relu = torch.nn.ReLU()
        self.conv1_bn = nn.BatchNorm2d(1)
        self.dense1_bn = nn.BatchNorm1d(20*75)

    def forward(self, x):
        x = self.relu(self.conv1(self.conv1_bn(x)))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.maxpool_time(x)
        xx = self.dropout(self.dense1_bn(torch.flatten(x, start_dim=1)))
        xx = self.dropout(self.relu(self.fc1(xx)))
        xx = F.softmax(self.fc2(xx),dim=1)
        return xx

class EEG_CNN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, 2)
        self.down2 = Down(128, 256, 1)
        self.down3 = Down(256, 512, 1)
        self.down4 = Down(512, 512, 1)
        factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(768, 512 // factor, bilinear)
        # self.up2 = Up(768, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.fc1 = nn.Linear(512*94, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(p=0.35)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # if len(x.shape)==3:
        #     x = x.unsqueeze(dim=-1).permute(0, 3, 1, 2)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        xx = self.dropout(torch.flatten(x5, start_dim=1))
        xx = self.relu(self.fc1(xx))
        xx = self.dropout(xx)
        xx = F.log_softmax(self.fc2(xx),dim=1)
        return xx

class EEG_CNN_2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.conv1 =  Eeg_Conv(1,9,kernel_size=(9,1),padding=(0,0))
        self.conv2 =  Eeg_Conv(1,8,kernel_size=(1,64),padding=(0,32))
        self.conv3 =  Eeg_Conv(8,8,kernel_size=(1,64),padding=(0,32))
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 8),stride=(1,8))
        self.fc1 = nn.Linear(8*9*23, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # if len(x.shape)==3:
        #     x = x.unsqueeze(dim=-1)
        x1 = self.conv1(x)
        x2 = x1.permute(0, 2, 1, 3)
        x3 = self.maxpool(self.conv2(x2))
        x4 = self.maxpool(self.conv3(x3))
        xx = self.dropout(torch.flatten(x4, start_dim=1))
        xx = self.relu(self.fc1(xx))
        xx = self.dropout(xx)
        xx = F.log_softmax(self.fc2(xx),dim=1)
        return xx

class EEG_CNN_3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.conv1 =  Eeg_Conv(1,9,kernel_size=(9,1),padding=(0,0))
        self.conv2 =  Eeg_Conv(1,8,kernel_size=(1,64),padding=(0,32))
        self.conv3 =  Eeg_Conv(8,8,kernel_size=(1,64),padding=(0,32))
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 8),stride=(1,8))
        self.fc1 = nn.Linear(8*9*23*3, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.relu = torch.nn.ReLU()



    def forward(self, xt1, xt2, xt3):

        x1 = self.conv1(xt1)
        x2 = x1.permute(0, 2, 1, 3)
        x3 = self.maxpool(self.conv2(x2))
        x4 = self.maxpool(self.conv3(x3))
        xo1 = torch.flatten(x4, start_dim=1)

        x1 = self.conv1(xt2)
        x2 = x1.permute(0, 2, 1, 3)
        x3 = self.maxpool(self.conv2(x2))
        x4 = self.maxpool(self.conv3(x3))
        xo2 = torch.flatten(x4, start_dim=1)

        x1 = self.conv1(xt3)
        x2 = x1.permute(0, 2, 1, 3)
        x3 = self.maxpool(self.conv2(x2))
        x4 = self.maxpool(self.conv3(x3))
        xo3 = torch.flatten(x4, start_dim=1)

        xx = self.dropout(torch.cat((xo1, xo2, xo3),dim=1))
        xx = self.relu(self.fc1(xx))
        xx = self.dropout(xx)
        xx = F.log_softmax(self.fc2(xx),dim=1)
        return xx

class EEG_CNN_4(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.conv1 =  Eeg_Conv(1,10,kernel_size=(2,30),padding=(0,15))
        self.conv2 =  Eeg_Conv(10,20,kernel_size=(1,15),padding=(0,7))
        self.conv3 =  Eeg_Conv(20,20,kernel_size=(4,2),padding=(0,0),stride=(1,2))
        # self.conv4 =  Eeg_Conv(20,20,kernel_size=(4,1),padding=(0,0))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(20*375, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # if len(x.shape)==3:
        #     x = x.unsqueeze(dim=-1)
        x1 = self.conv1(x)
        x3 = self.maxpool(self.conv2(x1))
        x4 = self.conv3(x3)
        # x5 = self.maxpool(self.conv4(x4))
        # print(x5.shape)
        xx = self.dropout(torch.flatten(x4, start_dim=1))
        xx = self.relu(self.fc1(xx))
        xx = self.dropout(xx)
        xx = F.log_softmax(self.fc2(xx),dim=1)
        return xx

class EEG_CNN_5(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.conv1 =  Eeg_Conv(1,10,kernel_size=(1,15),padding=(0,7))
        self.conv2 =  Eeg_Conv(10,20,kernel_size=(1,10),padding=(0,5))
        self.conv3 =  Eeg_Conv(20,20,kernel_size=(1,10),padding=(0,5))
        self.conv4 =  Eeg_Conv(20,20,kernel_size=(4,1),padding=(0,0),stride=(1,1))
        # self.conv4 =  Eeg_Conv(20,20,kernel_size=(4,1),padding=(0,0))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool_time = nn.MaxPool2d(kernel_size=(1, 5))
        self.fc1 = nn.Linear(20*90, 2)
        # self.fc2 = nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(p=0.50)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool_time(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        xx = self.dropout(torch.flatten(x, start_dim=1))
        # xx = self.relu(self.fc1(xx))
        # xx = self.dropout(xx)
        xx = F.log_softmax(self.fc1(xx),dim=1)
        return xx



class EEG_UCnet_1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(2,10), padding=(1,5), stride=(1,1)),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 10, kernel_size=(2,10), padding=(0,4), stride=(1,1)),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=(1, 5), padding=(0, 2), stride=(1, 1)),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 20, kernel_size=(1, 5), padding=(0, 2), stride=(1, 1)),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=(4, 1), padding=(0, 0), stride=(1, 1)),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 20, kernel_size=1, padding=(0, 0), stride=(1, 1)),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True)
        )
        # self.conv2 =  DoubleConv(10,20,kernel_size=(1,5),padding=(0,2),stride=(1,1))
        # self.conv3 =  DoubleConv(20,20,kernel_size=(4,1),padding=(0,0),stride=(1,1))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool_time = nn.MaxPool2d(kernel_size=(1, 6))
        self.fc1 = nn.Linear(20*75, 10)
        self.fc2 = nn.Linear(10, 2)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.relu = torch.nn.ReLU()
        self.conv1_bn = nn.BatchNorm2d(1)
        self.dense1_bn = nn.BatchNorm1d(20*75)

        self.up1 = nn.ConvTranspose2d(20 , 20, kernel_size=(1,6), stride=(1,6))
        self.up2 = nn.ConvTranspose2d(20 , 20, kernel_size=(4,1), stride=(1,1))
        self.up3 = nn.ConvTranspose2d(10 , 10, kernel_size=(2,2), stride=(2,2))

        self.outconv1 = DoubleConv(40, 20)
        self.outconv2 = DoubleConv(40, 10)
        self.outconv3 = DoubleConv(20, 1)
        self.outconv4 = DoubleConv(2, 1)


    def forward(self, x):
        x1 = self.relu(self.conv1(self.conv1_bn(x)))
        x2 = self.maxpool(x1)
        x3 = self.relu(self.conv2(x2))
        x4 = self.relu(self.conv3(x3))
        x5 = self.maxpool_time(x4)
        x6 = self.up1(x5)
        x6d = self.outconv1(torch.cat([x6, x4], dim=1))
        x7 = self.up2(x6d)
        x7d = self.outconv2(torch.cat([x7, x3], dim=1))
        x8 = self.up3(x7d)
        x8d = self.outconv3(torch.cat([x8, x1], dim=1))
        x9 = self.outconv4(torch.cat([x8d, x], dim=1))

        xx = self.dropout(self.dense1_bn(torch.flatten(x5, start_dim=1)))
        xx = self.dropout(self.relu(self.fc1(xx)))
        xx = F.softmax(self.fc2(xx),dim=1)
        return xx, x9

class EEG_UCnet_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EEG_Encoder()
        self.up1 = nn.ConvTranspose2d(20 , 20, kernel_size=(1,6), stride=(1,6))
        self.up2 = nn.ConvTranspose2d(20 , 20, kernel_size=(4,1), stride=(1,1))
        self.up3 = nn.ConvTranspose2d(10 , 10, kernel_size=(2,2), stride=(2,2))

        self.outconv1 = DoubleConv(40, 20)
        self.outconv2 = DoubleConv(40, 10)
        self.outconv3 = DoubleConv(20, 1)
        self.outconv4 = DoubleConv(2, 1)

        self.dense1_bn = nn.BatchNorm1d(20*75)
        self.fc1 = nn.Linear(20*75, 10)
        self.fc2 = nn.Linear(10, 2)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.relu = torch.nn.ReLU()


    def forward(self,x):
        x5, x4, x3, x1 = self.encoder(x)
        x6 = self.up1(x5)
        x6d = self.outconv1(torch.cat([x6, x4], dim=1))
        x7 = self.up2(x6d)
        x7d = self.outconv2(torch.cat([x7, x3], dim=1))
        x8 = self.up3(x7d)
        x8d = self.outconv3(torch.cat([x8, x1], dim=1))
        x9 = self.outconv4(torch.cat([x8d, x], dim=1))

        xx = self.dropout(self.dense1_bn(torch.flatten(x5, start_dim=1)))
        xx = self.dropout(self.relu(self.fc1(xx)))
        xx = F.softmax(self.fc2(xx), dim=1)
        return xx, x9

class EEG_Unet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.encoder = EEG_Encoder(n)
        self.up1 = nn.ConvTranspose2d(20*n , 20*n, kernel_size=(1,6), stride=(1,6))
        self.up2 = nn.ConvTranspose2d(20*n , 20*n, kernel_size=(4,1), stride=(1,1))
        self.up3 = nn.ConvTranspose2d(int(20*(n/2)) , int(20*(n/2)), kernel_size=(2,2), stride=(2,2))

        # self.outconv1 = DoubleConv(40, 20)
        # self.outconv2 = DoubleConv(40, 10)
        # self.outconv3 = DoubleConv(20, 1)
        # self.outconv4 = DoubleConv(2, 1)

        self.outconv1 = nn.Sequential(
            nn.Conv2d(20*(n*2), 20*n, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(20*n),
            nn.Tanh(),
            nn.Conv2d(20*n, 20*n, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(20*n),
            nn.Tanh()
        )
        self.outconv2 = nn.Sequential(
            nn.Conv2d(int(20*(n*2)), int(20*(n/2)), kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(int(20*(n/2))),
            nn.Tanh(),
            nn.Conv2d(int(20*(n/2)), int(20*(n/2)), kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(int(20*(n/2))),
            nn.Tanh()
        )
        self.outconv3 = nn.Sequential(
            nn.Conv2d(int(20*(n/2)), 1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1)

        )


    def forward(self,x):
        x5, x4, x3, x2 = self.encoder(x)
        x6 = self.outconv1(torch.cat([self.up1(x5), x4], dim=1))
        x7 = self.outconv2(torch.cat([self.up2(x6), x3], dim=1))
        x8= self.outconv3(self.up3(x7))
        return x8

class EEG_Encoder(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.pad_1 = nn.ReflectionPad2d((5,5,1,1))
        self.conv1 = nn.Conv2d(1, 10*dec, kernel_size=(2, 10), stride=(1, 1))
        self.pad_2 = nn.ReflectionPad2d((2,2,0,0))
        self.conv2 = nn.Conv2d(10*dec, 20*dec, kernel_size=(1, 5), stride=(1, 1))
        self.conv3 = nn.Conv2d(20*dec, 20*dec, kernel_size=(4, 1), padding=(0, 0), stride=(1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool_time = nn.MaxPool2d(kernel_size=(1, 6))
        self.relu = torch.nn.ReLU()
        self.conv1_bn = nn.BatchNorm2d(1)

    def forward(self,x):
        x1 = self.relu(self.conv1(self.pad_1(self.conv1_bn(x))))
        x2 = self.maxpool(x1)
        x3 = self.relu(self.conv2(self.pad_2(x2)))
        x4 = self.relu(self.conv3(x3))
        x5 = self.maxpool_time(x4)
        return x5

class STFT_Encoder(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 10*dec, kernel_size=(2, 5, 5), padding=(1, 2, 2), stride=(1, 1, 1))
        self.conv2 = nn.Conv3d(10*dec, 20*dec, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1))
        self.conv3 = nn.Conv3d(20*dec, 20*dec, kernel_size=(4, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.maxpool_timefreq = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.relu = torch.nn.ReLU()
        self.conv1_bn = nn.BatchNorm3d(1)

    def forward(self,x):
        x1 = self.relu(self.conv1(self.conv1_bn(x)))
        x2 = self.maxpool(x1)
        x3 = self.relu(self.conv2(x2))
        x4 = self.relu(self.conv3(x3))
        x5 = self.maxpool_timefreq(x4)
        return x5

class Eeg_Conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=(3,3), padding=(1,1), stride = (1,1)):
        super().__init__()
        self.eeg_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.eeg_conv(x)



class STFT_CNN_x(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(768, 512 // factor, bilinear)
        self.up2 = Up(768, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.fc1 = nn.Linear(512*2*4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x1 =
        x1 = self.inc(x.float())
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xx = self.fc1(torch.flatten(x4, start_dim=1))
        xx = self.dropout(xx)
        xx = F.log_softmax(self.fc2(xx),dim=1)

        return xx



class Encoder_block(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

    def forward(self, x):
        x1 = self.inc(x.float())
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x4

class Decoder_block(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

    def forward(self, x):
        x1 = self.inc(x.float())
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x4

class Up_to_EEG(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)