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
    

class MultiHeadAttention(nn.Module):

    def __init__(self,d_model, head ):
        super().__init__()

        assert d_model % head == 0
        
        self.head= head

        self.head_dim= d_model // head

        self.q_proj= nn.Linear(d_model,d_model)
        self.k_proj= nn.Linear(d_model,d_model)
        self.v_proj= nn.Linear(d_model,d_model)
        self.Wo_proj= nn.Linear(d_model, d_model)


    def forward(self,x, mask=None):

        B,S,D= x.shape

        Q= self.q_proj(x)
        K= self.k_proj(x)
        V= self.v_proj(x)

        Q= Q.view(B,S,self.head, self.head_dim).permute(0,2,1,3)
        K= K.view(B,S,self.head, self.head_dim).permute(0,2,1,3)
        V= V.view(B,S,self.head, self.head_dim).permute(0,2,1,3)

        score= Q @ K.permute(0,1,3,2)
        score= score/(self.head_dim ** 0.5)

        if mask is not None:
            score= score.masked_fill(mask==0, float("-inf"))

        attn= torch.softmax(score, dim=-1)

        out= attn @ V

        out= out.permute(0,2,1,3).contagious().view(B,S,D)

        out= self.Wo_proj(out)

        return out 















    