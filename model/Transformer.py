import torch.nn as nn
import torch
from .Transformer_block import encoder_block,decoder_block

class encoder_stack(nn.Module):

    def __init__(self, layers):
        super().__init__()

        self.layers= nn.ModuleList(layers)

    
    def forward(self,x):

        for layer in self.layers:
            x= layer(x)

        return x 
    

class decoder_stack(nn.Module):

    def __init__(self, layers):
        super().__init__()

        self.layers= nn.ModuleList(layers)

    
    def forward(self,x, enc_out):

        for layer in self.layers:
            x= layer(x,enc_out)

        return x 


class Transformer(nn.Module):
    
    def __init__(self, encoder, decoder, embed_dim,src_vocab_size,tgt_vocab_size):
        super().__init__()

        self.src_embedding= nn.Embedding(src_vocab_size,embed_dim)
        self.tgt_embedding= nn.Embedding(tgt_vocab_size,embed_dim)

        self.fc_out= nn.Linear(embed_dim, tgt_vocab_size)

        self.encoder= encoder
        self.decoder= decoder 

    def forward(self,src,trgt):

        src= self.src_embedding(src)
        trgt= self.tgt_embedding(trgt)

        enc_out= self.encoder(src)
        dec_out= self.decoder(trgt,enc_out)

        return self.fc_out(dec_out)
    