
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.nn.utils.rnn as rnn


class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, embed_dim=512, k=30):
        super().__init__()

        self.tanh = nn.Tanh()

        self.W_b = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_v = nn.Parameter(torch.randn(k, embed_dim))
        self.W_q = nn.Parameter(torch.randn(k, embed_dim))
        self.w_hv = nn.Parameter(torch.randn(k, 1))
        self.w_hq = nn.Parameter(torch.randn(k, 1))


    def forward(self, V, Q):  # V : B x embed_dim x M, Q : B x L x embed_dim
        M = V.shape[2]
        L = Q.shape[1]

        C = torch.matmul(Q, torch.matmul(self.W_b, V)) # B x L x M

        H_v = self.tanh(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))                            # B x k x M
        H_q = self.tanh(torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))           # B x k x L

        #a_v = torch.squeeze(fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)) # B x M
        #a_q = torch.squeeze(fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)) # B x L

        a_v = fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2) # B x 1 x M
        a_q = fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2) # B x 1 x L

        a_v = torch.stack([a_v]*M, dim=1).squeeze() # B x M x M
        a_q = torch.stack([a_q]*L, dim=1).squeeze() # B x L x L

        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1))) # B x embed_dim x M
        q = torch.squeeze(torch.matmul(a_q, Q))                  # B x embed_dim x L

        return v, q