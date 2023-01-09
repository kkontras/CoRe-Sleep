import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from graphs.models.custom_unet import PositionalEncoder, EEG_Encoder, STFT_Encoder
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = src.shape[0]
        trg_len = src.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).cuda()

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = (torch.ones([hidden[0].shape[0],2])*torch.Tensor([0.5]).float()).cuda()

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[:,t] = output.squeeze()

            input = output.squeeze()

        return outputs

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers


        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(1)

        # input = [1, batch size, emb dim]

        embedded = input
        # print(embedded.shape)
        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell

class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch size, src len, emb dim]

        embedded = self.dropout(src)

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell

class BiLSTM_ATT_BSeq(nn.Module):
    def __init__(self, encs=[None,None]):
        super().__init__()
        if encs[0]:
            self.encoder_eeg = encs[0]
        else:
            self.encoder_eeg = EEG_Encoder(2)
        if encs[1]:
            self.encoder_stft = encs[1]
        else:
            self.encoder_stft = STFT_Encoder(2)

        # Seq2Seq model
        # enc = Encoder(4000, 512, 2, 0.5)
        # dec = Decoder(2, 2, 512, 2, 0.5)
        #
        # self.seq2seq = Seq2Seq(enc, dec)

        #Attention
        self.pos = PositionalEncoder(d_model= 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.att = nn.TransformerEncoder(encoder_layer, num_layers= 6)
        self.hidden2tag = nn.Linear(256, 2)

        self.emb_fc = nn.Linear(4000,256)


        # self.fc1 = nn.Linear(128, 32)
        # self.fc2 = nn.Linear(32, 2)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.relu = torch.nn.ReLU()
        # self.dense1_bn = nn.BatchNorm1d(1024)
        # self.dense2_bn = nn.BatchNorm1d(128)

    def forward(self, x):

        seq_1 = x[0].flatten(start_dim = 0, end_dim=1)

        seq_2 = x[1].flatten(start_dim = 0, end_dim=1)

        x_eeg,_,_,_ = self.encoder_eeg(seq_1)
        x_stft,_,_,_ = self.encoder_stft(seq_2)

        xcat = torch.cat([x_eeg.flatten(start_dim=1), x_stft.flatten(start_dim=1)], dim=1)
        xcat = xcat.view([x[0].shape[0], x[0].shape[1], 4000])
        xcat = self.relu(self.emb_fc(self.dropout(xcat)))

        # out = self.dropout(self.dense1_bn(lstm_out.permute(1,0,2).contiguous().flatten(start_dim=1)))
        # out = self.dropout(self.dense1_bn(lstm_out[-1,:,:]))
        # out = self.dropout(self.dense2_bn(self.relu(self.fc1(out))))
        # out = F.softmax(self.fc2(out), dim=1)

        # Seq2Seq
        # a = xcat.view([x[0].shape[0], x[0].shape[1], 4000])
        # out = self.seq2seq(a)
        # return out

        # Attention approach
        att_out = self.att(self.pos(xcat.view([x[0].shape[0], 16, 256])))
        tag_space = self.hidden2tag(att_out.flatten(start_dim=1,end_dim=1)) # concat

        #
        tag_scores = F.log_softmax(tag_space, dim=1)
        # tag_scores = self.decoder(lstm_out,h)
        return tag_scores.view([x[0].shape[0], 16, 2])

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=16):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)