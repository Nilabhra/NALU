import torch
from torch import nn
from torch.nn import functional as F


class NAC(nn.Module):
    '''
    Class implementing Neural Accumulator (NAC)
    as described in https://arxiv.org/abs/1808.00508
    '''
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W_hat = nn.Parameter(torch.Tensor(self.out_dim, self.in_dim))
        self.M_hat = nn.Parameter(torch.Tensor(self.out_dim, self.in_dim))
        nn.init.xavier_normal_(self.W_hat)
        nn.init.xavier_normal_(self.M_hat)
        self.bias = None
        
    def forward(self, x):
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        return F.linear(x, W, self.bias)