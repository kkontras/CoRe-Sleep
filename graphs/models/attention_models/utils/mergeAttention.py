import torch.nn.functional as F
import torch.nn as nn
import torch

class Merge_Attention(nn.Module):
    def __init__(self, input_sizes, d_model):
        super().__init__()
        for i, input_size in enumerate(input_sizes):
            setattr(self, "emb_fc_{}".format(i), nn.Linear(int(input_size),d_model))
        self.emb_fc = lambda x, e_dim: [getattr(self,"emb_fc_{}".format(i))(xi.flatten(start_dim=1)).unsqueeze(dim=e_dim) for i, xi in enumerate(x)]
        self.alpha = torch.nn.Parameter(torch.rand(d_model),requires_grad=True)
        self.b = torch.nn.Parameter(torch.rand(1),requires_grad=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):

        extra_dimension = 2
        emb_x = self.emb_fc(x,extra_dimension)

        # Attention weights created from the common query alpha learnable value.
        xi_alpha = [self.softmax(self.alpha*emb_xi.squeeze()+ self.b).unsqueeze(dim=extra_dimension) for emb_xi in emb_x]

        #concat views/modalities
        emb_x = torch.cat(emb_x,dim=extra_dimension)
        xi_alpha = torch.cat(xi_alpha,dim=extra_dimension)

        # Weighted sum of those modalities/views
        x_concat = (xi_alpha*emb_x).sum(dim=extra_dimension)

        return x_concat

    def merge_dimensions(self, x):
        # x -> batch x dimension_to_be_merged x [features]
        extra_dimension = 2
        emb_x = [self.emb_fc[i](x.flatten(start_dim=2)[:,i,:]).unsqueeze(dim=extra_dimension) for i in range(x.shape[-2])]
        xi_alpha = [F.softmax(self.alpha*emb_xi.squeeze()+ self.b,dim=1).unsqueeze(dim=extra_dimension) for emb_xi in emb_x]

        emb_x = torch.cat(emb_x,dim=extra_dimension)
        xi_alpha = torch.cat(xi_alpha,dim=extra_dimension)

        x_concat = (xi_alpha*emb_x).sum(dim=extra_dimension)# Weighted sum

        return x_concat


class Center_Alligned_Attention(nn.Module):
    def __init__(self, d_model, n_heads, device):
        super().__init__()
        self.att = nn.MultiheadAttention(d_model, n_heads).to(device)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        assert  x.shape[0]%2 == 1, "Big window must be odd int"
        mid = x.shape[0]//2
        attn_output, attn_output_weights = self.att (x[mid].unsqueeze(dim=0), x, x)
        return attn_output.squeeze()