import torch
from torch import nn
from models.nac import NAC


class NALU(nn.Module):
    '''
    Class implementing Neural Arithmetic Logic Unit (NALU)
    with a small deviation from the original one described
    here: https://arxiv.org/abs/1808.00508
    '''
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.G = nn.Parameter(torch.Tensor(1, 1))
        nn.init.xavier_normal_(self.G)
        self.nac = NAC(self.in_dim, self.out_dim)
        self.eps = 1e-12

    def forward(self, x):
        a = self.nac(x)
        g = torch.sigmoid(self.G)
        m = self.nac(torch.log(torch.abs(x) + self.eps))
        m = torch.exp(m)
        y = (g * a) + (1 - g) * m
        return y