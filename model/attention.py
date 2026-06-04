import torch.nn as nn
import torch

class SelfAttention(nn.Module):

    def __init__(self,d_model):
        super().__init__()

        self.q_proj= nn.Linear(d_model,d_model)
        self.k_proj= nn.Linear(d_model,d_model)
        self.v_proj= nn.Linear(d_model,d_model)

        self.scale= d_model ** 0.5

    def forward(self, x, mask=None):

        Q= self.q_proj(x)
        K= self.k_proj(x)
        V= self.v_proj(x)

        scores= torch.matmul(Q, K.transpose(-2,-1))
        scores= (scores)/self.scale

        if mask is not None:
            scores= scores.masked_fill(mask==0, float("-inf"))

        attn= torch.softmax(scores, dim=-1)

        out= torch.matmul(attn,V)

        return out 






    