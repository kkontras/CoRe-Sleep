import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init

from graphs.models.xsleepnet.filterbank_shape import FilterbankShape
import einops
import math
import torch.functional as F

class F1_conv(nn.Module):
    def __init__(self, args):
        super().__init__()

        in_channels = args.in_channels if "in_channels" in args else 1
        dropout = args.cnn_dropout if "cnn_dropout" in args else 1
        kernel_size = args.kernel_size if "kernel_size" in args else 31
        stride_step = 2
        padding = 15

        self.conv = nn.Sequential(

            nn.Conv1d( in_channels= in_channels, out_channels= 16, kernel_size=kernel_size, stride=stride_step, padding=padding),
            # nn.MaxPool1d(kernel_size=2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=kernel_size, stride=stride_step, padding=padding),
            # nn.MaxPool1d(kernel_size=2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=stride_step, padding=padding),
            # nn.MaxPool1d(kernel_size=2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=stride_step, padding=padding),
            # nn.MaxPool1d(kernel_size=2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride_step, padding=padding),
            # nn.MaxPool1d(kernel_size=2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=stride_step, padding=padding),
            # nn.MaxPool1d(kernel_size=2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride_step, padding=padding),
            # nn.MaxPool1d(kernel_size=2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=stride_step, padding=padding),
            # nn.MaxPool1d(kernel_size=2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=stride_step, padding=padding),
            # nn.MaxPool1d(kernel_size=2),
            nn.PReLU()
        )

    def forward(self, x_time):
        output = self.conv(x_time)
        output = output.flatten(start_dim=1)
        return {"conv_features": output}

class F2_BNLSTM_Filterbank(nn.Module):
    def __init__(self, args):
        super().__init__()

        # self.learnable_filters = torch.nn.Parameter(torch.randn(129, 32), requires_grad=True)
        filter_bank_filters = FilterbankShape().lin_tri_filter_shape(   nfilt=args.seq_nfilter,
                                                                        nfft=args.seq_nfft,
                                                                        samplerate=args.seq_samplerate,
                                                                        lowfreq=args.seq_lowfreq,
                                                                        highfreq=args.seq_highfreq)

        self.learnable_filters = torch.nn.Parameter(torch.randn(129, args.seq_nfilter) * torch.from_numpy(filter_bank_filters).float(), requires_grad=True)

        # self.bnlstm = LSTM(cell_class=BNLSTMCell, input_size = args.seq_nfilter, hidden_size=args.seq_nhidden1, max_length=29)

        # self.bnlstm = torch.nn.LSTM(input_size = args.seq_nfilter, hidden_size=args.seq_nfilter, batch_first=True, bidirectional=True)
        self.bnlstm = BNLSTM(input_size = args.seq_nfilter, hidden_size=args.seq_nfilter, batch_first=True, bidirectional=True)

        self.att_aggr = Attention_Aggregator(att_size=args.seq_attention_size1, dmodel=args.seq_nfilter*2)

    def forward(self, x_stft, return_att_weights = False):
        x_stft = torch.einsum("bsf, fd -> bsd", x_stft, self.learnable_filters)
        x_stft, _ = self.bnlstm(x_stft)
        # x_stft = einops.rearrange(x_stft, "inner b features -> b inner features")

        x_stft = self.att_aggr(x_stft)
        output = {"f2_stft_output":x_stft["att_aggr_features"]}
        if return_att_weights:
            output.update({"f2_att_weights":x_stft["att_weights"]})
        return output
class Attention_Aggregator(nn.Module):
    def __init__(self, att_size, dmodel):
        super().__init__()

        self.Wa = nn.Parameter(torch.randn(att_size,dmodel), requires_grad=True)
        self.ba = nn.Parameter(torch.randn(att_size), requires_grad=True)
        self.ae = nn.Parameter(torch.randn(att_size), requires_grad=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input):
        """
        Attentional Aggregator for RNNs with bias and output linear projection ae.

        :param input: [batch sequence features]
        :return: att_aggr_features = [batch sequence features] attended
                att_weights = [batch sequence] attention weights .sum(dim=1) equals 1
        """
        att_weights = torch.tanh(torch.einsum("bsf, af -> bsa", input, self.Wa) + self.ba)
        att_weights = torch.einsum("bsa, a -> bs", att_weights, self.ae)
        att_weights = self.softmax(att_weights)
        output = torch.einsum("bsf, bs -> bsf", input, att_weights).sum(dim=1)
        return {"att_aggr_features": output, "att_weights": att_weights}

class XSleepnet(nn.Module):
    def __init__(self, args, encs=[None]):
        super().__init__()
        self.encs = encs
        self.args = args

        self.f1_conv = F1_conv(args=args)

        self.f2_bnsltm_filterbank = F2_BNLSTM_Filterbank(args)

        # self.time_gru = LSTM(cell_class=nn.GRUCell, input_size = 1536, hidden_size=args.seq_nhidden1)
        # self.stft_lstm = LSTM(cell_class=BNLSTMCell, input_size = args.seq_nhidden1, hidden_size=args.seq_nhidden2, max_length=29)

        # self.time_gru = torch.nn.GRU(input_size = 1536, hidden_size=args.seq_nhidden1, batch_first=True, bidirectional=True)
        # self.stft_lstm = torch.nn.LSTM(input_size = args.seq_nhidden1, hidden_size=args.seq_nhidden1, batch_first=True, bidirectional=True)

        # self.time_gru = BNGRU(input_size = 1536, hidden_size=args.seq_nhidden1, batch_first=True, bidirectional=True)
        self.time_gru = BNLSTM(input_size = 1536, hidden_size=args.seq_nhidden1, batch_first=True, bidirectional=True)
        self.stft_lstm = BNLSTM(input_size = args.seq_nhidden1, hidden_size=args.seq_nhidden1, batch_first=True, bidirectional=True)

        # self.fc_out_time_0 = nn.Sequential(
        #     nn.Linear(1536,64),
        #     nn.ReLU()
        # )
        self.fc_out_time = nn.Linear(128,5)
        self.fc_out_stft = nn.Linear(128,5)
        self.fc_out = nn.Linear(256,5)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, **kwargs):
        x_time, x_stft = x["time_eeg"], x["stft_eeg"]


        x_batch, x_outer = x_time.shape[0], x_time.shape[1]

        x_time = einops.rearrange(x_time, "b outer time mod -> (b outer) mod time")
        f1_output = self.f1_conv(x_time)
        x_time = einops.rearrange(f1_output["conv_features"], "(b outer) features -> b outer features", b=x_batch, outer=x_outer)

        x_stft = einops.rearrange(x_stft, "b outer mod ch f inner -> (b outer) inner (mod ch f)")
        f2_output = self.f2_bnsltm_filterbank(x_stft)
        x_stft = einops.rearrange(f2_output["f2_stft_output"], "(b outer) f -> b outer f", b=x_batch, outer=x_outer)

        x_time, a = self.time_gru(x_time)
        x_stft, _ = self.stft_lstm(x_stft)

        x_time = einops.rearrange(x_time, " b outer f -> (b outer) f")
        x_stft = einops.rearrange(x_stft, " b outer f -> (b outer) f")
        # x_time = self.fc_out_time_0(x_time)


        comb_output = self.fc_out(torch.cat([x_time, x_stft], dim=1))
        x_time = self.fc_out_time(x_time)
        x_stft = self.fc_out_stft(x_stft)

        return {"preds":{"combined":comb_output,"time":x_time,"stft":x_stft}}


class SeparatedBatchNorm1d(nn.Module):
    """
    A batch normalization module which keeps its running mean
    and variance separately per timestep.
    """

    def __init__(self, num_features, max_length, eps=1e-5, momentum=0.1,
                 affine=True):
        """
        Most parts are copied from
        torch.nn.modules.batchnorm._BatchNorm.
        """

        super(SeparatedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(num_features))
            self.bias = nn.Parameter(torch.FloatTensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        for i in range(max_length):
            self.register_buffer(
                'running_mean_{}'.format(i), torch.zeros(num_features))
            self.register_buffer(
                'running_var_{}'.format(i), torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.max_length):
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i.zero_()
            running_var_i.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input_):
        if input_.size(1) != self.running_mean_0.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input_.size(1), self.num_features))

    def forward(self, input_, time):
        self._check_input_dim(input_)
        if time >= self.max_length:
            time = self.max_length - 1
        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))
        return functional.batch_norm(
            input=input_, running_mean=running_mean, running_var=running_var,
            weight=self.weight, bias=self.bias, training=self.training,
            momentum=self.momentum, eps=self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' max_length={max_length}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))

# class LSTMCell(nn.Module):
#     """A basic LSTM cell."""
#
#     def __init__(self, input_size, hidden_size, use_bias=True):
#         """
#         Most parts are copied from torch.nn.LSTMCell.
#         """
#
#         super(LSTMCell, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.use_bias = use_bias
#         self.weight_ih = nn.Parameter(
#             torch.FloatTensor(input_size, 4 * hidden_size))
#         self.weight_hh = nn.Parameter(
#             torch.FloatTensor(hidden_size, 4 * hidden_size))
#         if use_bias:
#             self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         """
#         Initialize parameters following the way proposed in the paper.
#         """
#
#         init.orthogonal_(self.weight_ih.data)
#         weight_hh_data = torch.eye(self.hidden_size)
#         weight_hh_data = weight_hh_data.repeat(1, 4)
#         self.weight_hh.data.set_(weight_hh_data)
#         # The bias is just set to zero vectors.
#         if self.use_bias:
#             init.constant_(self.bias.data, val=0)
#
#     def forward(self, input_, hx):
#         """
#         Args:
#             input_: A (batch, input_size) tensor containing input
#                 features.
#             hx: A tuple (h_0, c_0), which contains the initial hidden
#                 and cell state, where the size of both states is
#                 (batch, hidden_size).
#         Returns:
#             h_1, c_1: Tensors containing the next hidden and cell state.
#         """
#
#         h_0, c_0 = hx
#         batch_size = h_0.size(0)
#         bias_batch = (self.bias.unsqueeze(0)
#                       .expand(batch_size, *self.bias.size()))
#         wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
#         wi = torch.mm(input_, self.weight_ih)
#         f, i, o, g = torch.split(wh_b + wi, split_size_or_sections=self.hidden_size, dim=1)
#         c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
#         h_1 = torch.sigmoid(o) * torch.tanh(c_1)
#         return h_1, c_1
#
#     def __repr__(self):
#         s = '{name}({input_size}, {hidden_size})'
#         return s.format(name=self.__class__.__name__, **self.__dict__)

class BNLSTMCell(nn.Module):
    """A BN-LSTM cell."""

    def __init__(self, input_size, hidden_size, max_length, use_bias=True):

        super(BNLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        # BN parameters
        self.bn_ih = SeparatedBatchNorm1d(
            num_features=4 * hidden_size, max_length=max_length)
        self.bn_hh = SeparatedBatchNorm1d(
            num_features=4 * hidden_size, max_length=max_length)
        self.bn_c = SeparatedBatchNorm1d(
            num_features=hidden_size, max_length=max_length)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """

        # The input-to-hidden weight matrix is initialized orthogonally.
        init.orthogonal_(self.weight_ih.data)
        # The hidden-to-hidden weight matrix is initialized as an identity
        # matrix.
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        init.constant_(self.bias.data, val=0)
        # Initialization of BN parameters.
        self.bn_ih.reset_parameters()
        self.bn_hh.reset_parameters()
        self.bn_c.reset_parameters()
        self.bn_ih.bias.data.fill_(0)
        self.bn_hh.bias.data.fill_(0)
        self.bn_ih.weight.data.fill_(0.1)
        self.bn_hh.weight.data.fill_(0.1)
        self.bn_c.weight.data.fill_(0.1)

    def forward(self, input_, hx, time):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
            time: The current timestep value, which is used to
                get appropriate running statistics.
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh = torch.mm(h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        bn_wh = self.bn_hh(wh, time=time)
        bn_wi = self.bn_ih(wi, time=time)
        f, i, o, g = torch.split(bn_wh + bn_wi + bias_batch,
                                 split_size_or_sections=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(self.bn_c(c_1, time=time))
        return h_1, c_1

class LSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=True, dropout_input=0, dropout_output=0, **kwargs):
        super(LSTM, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout_input = dropout_input
        self.dropout_output = dropout_output

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer_input = nn.Dropout(dropout_input)
        self.dropout_layer_output = nn.Dropout(dropout_output)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            if isinstance(cell, BNLSTMCell):
                h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
            elif isinstance(cell, nn.GRUCell):
                h_next = cell(input_[time], hx[0])
                c_next = h_next
            else:
                raise NotImplementedError("We dont support this rnn cell")

            mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            h_next = h_next * mask + hx[0] * (1 - mask)
            c_next = c_next * mask + hx[1] * (1 - mask)
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size)).cuda()

        if hx is None:
            hx = (Variable(nn.init.xavier_uniform(torch.randn([self.num_layers, batch_size, self.hidden_size]))).cuda(),
                  Variable(nn.init.xavier_uniform(torch.randn([self.num_layers, batch_size, self.hidden_size]))).cuda())
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            hx_layer = (hx[0][layer, :, :], hx[1][layer, :, :])
            input_ = self.dropout_layer_input(input_)
            if layer == 0:
                layer_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(
                    cell=cell, input_=input_, length=length, hx=hx_layer)
            else:
                layer_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(
                    cell=cell, input_=layer_output, length=length, hx=hx_layer)

            input_ = self.dropout_layer_output(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        return output, (h_n, c_n)


class GRUCell(nn.Module):
    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.ih = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.hh = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.bn_ih = nn.BatchNorm1d(3 * hidden_size, affine=False)
        self.bn_hh = nn.BatchNorm1d(3 * hidden_size, affine=False)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx):
        input = input.view(-1, input.size(1))

        ih = self.ih(input)
        hh = self.hh(hx)

        bn_ih = self.bn_ih(ih)
        bn_hh = self.bn_hh(hh)

        gate_x = bn_ih.squeeze()
        gate_h = bn_hh.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = self.sigmoid(i_r + h_r)
        inputgate = self.sigmoid(i_i + h_i)
        newgate = self.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hx - newgate)

        return hy

class BNLSTMCell_2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BNLSTMCell_2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

        self.bn_ih = nn.BatchNorm1d(4 * self.hidden_size, affine=False)
        self.bn_hh = nn.BatchNorm1d(4 * self.hidden_size, affine=False)
        self.bn_c = nn.BatchNorm1d(self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.weight_hh.data[:, :self.hidden_size])
        nn.init.orthogonal_(self.weight_hh.data[:, self.hidden_size:2 * self.hidden_size])
        nn.init.orthogonal_(self.weight_hh.data[:, 2 * self.hidden_size:3 * self.hidden_size])
        nn.init.eye_(self.weight_hh.data[:, 3 * self.hidden_size:])
        self.weight_hh.data[:, 3 * self.hidden_size:] *= 0.95

    def forward(self, input, hx):
        h, c = hx
        ih = torch.matmul(input, self.weight_ih)
        hh = torch.matmul(h, self.weight_hh)
        bn_ih = self.bn_ih(ih)
        bn_hh = self.bn_hh(hh)
        hidden = bn_ih + bn_hh + self.bias

        i, f, o, g = torch.split(hidden, split_size_or_sections=self.hidden_size, dim=1)
        new_c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        new_h = torch.sigmoid(o) * torch.tanh(self.bn_c(new_c))
        return (new_h, new_c)


class BNGRU(nn.Module):
    # hellozgy /bnlstm-pytorch
    def __init__(self, input_size, hidden_size, batch_first=False, bidirectional=False):
        super(BNGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.lstm_f = GRUCell(input_size, hidden_size)
        if bidirectional:
            self.lstm_b = GRUCell(input_size, hidden_size)
        self.h0 = nn.Parameter(torch.Tensor(2 if self.bidirectional else 1, 1, self.hidden_size))
        # self.c0 = nn.Parameter(torch.Tensor(2 if self.bidirectional else 1, 1, self.hidden_size))
        nn.init.normal_(self.h0, mean=0, std=0.1)
        # nn.init.normal_(self.c0, mean=0, std=0.1)

    def forward(self, input, hx=None):
        if not self.batch_first:
            input = input.transpose(0, 1)
        batch_size, seq_len, dim = input.size()
        if hx:
            init_state = hx
        else:
            init_state = self.h0.repeat(1, batch_size, 1)

        hiddens_f = []
        final_hx_f = None
        hx = init_state[0]
        for i in range(seq_len):
            hx = self.lstm_f(input[:, i, :], hx)
            hiddens_f.append(hx)
            final_hx_f = hx
        hiddens_f = torch.stack(hiddens_f, 1)

        if self.bidirectional:
            hiddens_b = []
            final_hx_b = None
            hx = init_state[1]
            for i in range(seq_len - 1, -1, -1):
                hx = self.lstm_b(input[:, i, :], hx)
                hiddens_b.append(hx)
                final_hx_b = hx
            hiddens_b.reverse()
            hiddens_b = torch.stack(hiddens_b, 1)

        if self.bidirectional:
            hiddens = torch.cat([hiddens_f, hiddens_b], -1)
            hx = torch.stack([final_hx_f[0], final_hx_b[0]], 0)
        else:
            hiddens = hiddens_f
            hx = hx[0].unsqueeze(0)
        if not self.batch_first:
            hiddens = hiddens.transpose(0, 1)
        return hiddens, hx

class BNLSTM(nn.Module):
    # hellozgy /bnlstm-pytorch
    def __init__(self, input_size, hidden_size, batch_first=False, bidirectional=False):
        super(BNLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.lstm_f = BNLSTMCell_2(input_size, hidden_size)
        if bidirectional:
            self.lstm_b = BNLSTMCell_2(input_size, hidden_size)
        self.h0 = nn.Parameter(torch.Tensor(2 if self.bidirectional else 1, 1, self.hidden_size))
        self.c0 = nn.Parameter(torch.Tensor(2 if self.bidirectional else 1, 1, self.hidden_size))
        nn.init.normal_(self.h0, mean=0, std=0.1)
        nn.init.normal_(self.c0, mean=0, std=0.1)

    def forward(self, input, hx=None):
        if not self.batch_first:
            input = input.transpose(0, 1)
        batch_size, seq_len, dim = input.size()
        if hx:
            init_state = hx
        else:
            init_state = (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1))

        hiddens_f = []
        final_hx_f = None
        hx = (init_state[0][0], init_state[1][0])
        for i in range(seq_len):
            hx = self.lstm_f(input[:, i, :], hx)
            hiddens_f.append(hx[0])
            final_hx_f = hx
        hiddens_f = torch.stack(hiddens_f, 1)

        if self.bidirectional:
            hiddens_b = []
            final_hx_b = None
            hx = (init_state[0][1], init_state[1][1])
            for i in range(seq_len - 1, -1, -1):
                hx = self.lstm_b(input[:, i, :], hx)
                hiddens_b.append(hx[0])
                final_hx_b = hx
            hiddens_b.reverse()
            hiddens_b = torch.stack(hiddens_b, 1)

        if self.bidirectional:
            hiddens = torch.cat([hiddens_f, hiddens_b], -1)
            hx = (torch.stack([final_hx_f[0], final_hx_b[0]], 0), torch.stack([final_hx_f[1], final_hx_b[1]], 0))
        else:
            hiddens = hiddens_f
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(1))
        if not self.batch_first:
            hiddens = hiddens.transpose(0, 1)
        return hiddens, hx