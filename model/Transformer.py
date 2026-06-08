import torch.nn as nn
import torch
from Transformer_block import encoder_block,decoder_block

class encoder_stack(nn.Module):

    def __init__(self, layers):
        super().__init__()

        self.layers= layers

    
    def forward(self,x):

        for layer in self.layers:
            x= layer(x)

        return x 
    

class decoder_stack(nn.Module):

    def __init__(self, layers):
        super().__init__()

        self.layers= layers

    
    def forward(self,x, enc_out):

        for layer in self.layers:
            x= layer(x,enc_out)

        return x 


class Transformer(nn.Module):
    
    