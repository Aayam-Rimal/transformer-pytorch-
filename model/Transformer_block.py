import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        self.gamma= nn.Parameter(torch.ones(d_model))
        self.beta= nn.Parameter(torch.zeros(d_model))
        self.eps= eps


    def forward(self, x):

        mean= x.mean(dim=-1, keepdim=True)
        var= x.var(dim=-1, keepdim=True, unbiased=False)

        x_norm= (x-mean)/torch.sqrt(var + self.eps)

        return self.gamma * x_norm + self.beta
    

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.W1= nn.Linear(d_model,d_ff)

        self.W2= nn.Linear(d_ff, d_model)

    def forward(self,x ):

        return self.W2(torch.relu(self.W1(x)))