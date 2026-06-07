import torch.nn as nn
import torch
from attention import MultiHeadAttention


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
    

class encoder_block(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, eps=1e-5):
        super().__init__()

        self.ffn1= FFN(d_model,d_ff)
        self.ln1= LayerNorm(d_model,eps=1e-5)
        self.ln2= LayerNorm(d_model,eps=1e-5)
        self.MHA= MultiHeadAttention(d_model, num_heads)

    def forward(self, src):

        z= self.MHA(src, mask=None)
        residual1= z + src

        ln1= self.ln1(residual1)
        ffn1= self.ffn1.forward(ln1)

        residual2= ffn1 + residual1
        ln2= self.ln2.forward(residual2)

        return ln2
    

    


