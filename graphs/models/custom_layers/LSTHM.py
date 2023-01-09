import torch
import time
from torch import nn
import torch.nn.functional as F


class LSTHM(nn.Module):

    def __init__(self, cell_size, in_size, hybrid_in_size):
        super(LSTHM, self).__init__()
        self.cell_size = cell_size
        self.in_size = in_size
        self.W = nn.Linear(in_size, 4 * self.cell_size)
        self.U = nn.Linear(cell_size, 4 * self.cell_size)
        self.V = nn.Linear(hybrid_in_size, 4 * self.cell_size)

    def step(self, x, ctm1, htm1, ztm1):

        input_affine = self.W(x)
        output_affine = self.U(htm1) if htm1 != None else torch.zeros(input_affine.shape).to(x.device)
        hybrid_affine = self.V(ztm1) if ztm1 != None else torch.zeros(input_affine.shape).to(x.device)

        sums = input_affine + output_affine + hybrid_affine

        # biases are already part of W and U and V
        f_t = F.sigmoid(sums[:, :self.cell_size])
        i_t = F.sigmoid(sums[:, self.cell_size:2 * self.cell_size])
        o_t = F.sigmoid(sums[:, 2 * self.cell_size:3 * self.cell_size])
        ch_t = F.tanh(sums[:, 3 * self.cell_size:])

        ctm1 = ctm1 if ctm1 != None else torch.zeros(f_t.shape).to(x.device)

        c_t = f_t * ctm1 + i_t * ch_t
        h_t = F.tanh(c_t) * o_t
        return c_t, h_t


class MultipleAttentionFusion(nn.Module):

    def __init__(self, attention_model, dim_reduce_nets, num_atts):
        super(MultipleAttentionFusion, self).__init__()
        self.attention_model = attention_model
        self.dim_reduce_nets = dim_reduce_nets
        self.num_atts = num_atts
        self.softmax = nn.Softmax(dim=1)

    def __call__(self, in_modalities):
        return self.fusion(in_modalities)

    def fusion(self, in_modalities):
        # getting some simple integers out
        num_modalities = len(in_modalities)
        # simply the tensor that goes into attention_model
        in_tensor = torch.cat(in_modalities, dim=1)
        # calculating attentions
        atts = self.softmax(self.attention_model(in_tensor).view([in_tensor.shape[0],self.num_atts,in_tensor.shape[1]])).flatten(start_dim=1, end_dim=2)
        # calculating the tensor that will be multiplied with the attention
        out_tensor = torch.cat([in_modalities[i].repeat(1, self.num_atts) for i in range(num_modalities)], dim=1)
        # calculating the attention
        att_out = atts * out_tensor

        # now to apply the dim_reduce networks
        # first back to however modalities were in the problem
        start = 0
        out_modalities = []
        for i in range(num_modalities):
            modality_length = in_modalities[i].shape[1] * self.num_atts
            out_modalities.append(att_out[:, start:start + modality_length])
            start = start + modality_length
        # apply the dim_reduce
        dim_reduced = [self.dim_reduce_nets[i](out_modalities[i]) for i in range(num_modalities)]
        # multiple attention done :)
        return dim_reduced, out_modalities

    def forward(self, x):
        print("Not yet implemented for nn.Sequential")
        exit(-1)


class MARN(nn.Module):
    def __init__(self):
        super(MARN, self).__init__()
        self.lsthm = LSTHM(128,128,128)
        attention_model = nn.Sequential(
            nn.Linear(256,128),
            nn.Tanh(),
            nn.Linear(128, 256*4),
        )
        dim_reduce_nets_eeg = nn.Sequential(
            nn.Linear(512,128),
            nn.Tanh(),
            nn.Linear(128,128),
        ).cuda()
        dim_reduce_nets_eog = nn.Sequential(
            nn.Linear(512,128),
            nn.Tanh(),
            nn.Linear(128,128),
        ).cuda()

        self.mab = MultipleAttentionFusion(attention_model, [dim_reduce_nets_eeg, dim_reduce_nets_eog], 4)
        self.co, self.ho, self.zo = [None, None], [None, None], [None, None]
        self.avg = nn.AvgPool1d(11)

    def step(self, eeg, eog, ho, co, zo):
        c_t_0, h_t_0 = self.lsthm.step(eeg, co[0], ho[0], zo[0])
        c_t_1, h_t_1 = self.lsthm.step(eog, co[1], ho[1], zo[1])
        h_t = [h_t_0, h_t_1]
        # c_t = torch.cat([c_t_0, c_t_1],dim=1)
        z_t, outs = self.mab.fusion(h_t)
        ho = h_t
        co = [c_t_0, c_t_1]
        zo = z_t
        return z_t, ho, co, zo

    def forward(self, eeg, eog):
        outs = []
        co, ho, zo = [None, None], [None, None], [None, None]
        #For on the time-steps. eeg and eog are batch x seq x features
        for i in range(eeg.shape[0]):
            out, ho, co, zo = self.step(eeg[i],eog[i], ho, co, zo)
            outs.append(torch.cat(out,dim=1).unsqueeze(dim=0))
        outs = torch.cat(outs,dim=0)
        # print(outs.shape)
        # outs = self.avg(outs)

        return outs

