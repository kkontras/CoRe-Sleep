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

class EP_EEG_CNN(nn.Module):
    def __init__(self, dec, _):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(18, 64 * dec, kernel_size=(5, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 1, 1)),
            nn.Conv2d(64 * dec, 128 * dec, kernel_size=(5, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(128 * dec, 16 * dec, kernel_size=(5, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((30, 1))  # 56
        )

    def forward(self, x):

        return self.conv(x)