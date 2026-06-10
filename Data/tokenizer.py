import torch.nn as nn
import torch
import json
import re 

class Tokenizer:

    def __init__(self, vocab_path):
        with open(vocab_path,"r") as f:
            self.word2id= json.load(f)

    
    def tokenizer(self,text):
        
        text= text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text=  text.split()

        return text 

    
    def encode(self, text):

        tokens= self.tokenizer(text)

        return [self.word2id.get(token, self.word2id["<unk>"]) for token in tokens]
       

    def decode(self,ids):

        inverted= {v:k for k,v in self.word2id.items()}

        return " ".join([inverted.get(i, "<unk>") for i in ids])
    


if __name__=="__main__":

    text= "Hello my namE is italy"

    tokenizer= Tokenizer("src_vocab.json")
    tokenized_text= tokenizer.encode(text)

    print(tokenized_text)

        
