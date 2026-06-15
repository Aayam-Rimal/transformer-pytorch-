from .tokenizer import Tokenizer
import json


class Dataset:

    def __init__(self, data_pair,src_vocab_path,tgt_vocab_path,src_vocab,tgt_vocab):

        self.data_pair= data_pair
        self.src_vocab= src_vocab
        self.tgt_vocab= tgt_vocab
        self.src_tokenize= Tokenizer(src_vocab_path)
        self.tgt_tokenize= Tokenizer(tgt_vocab_path)

    def __len__(self):
        return len(self.data_pair)
    

    def __getitem__(self, idx):

        src,trgt= self.data_pair[idx]

        src_id= self.src_tokenize.encode(src)
        tgt_id= self.tgt_tokenize.encode(trgt)

        tgt_input_id= [self.tgt_vocab["<bos>"]] +  tgt_id
        tgt_output_id= tgt_id + [self.tgt_vocab["<eos>"]]

        return src_id,tgt_input_id,tgt_output_id
    


    
if __name__=="__main__":

    data_pair = [
    ("hello world", "ciao mondo"),
    ("how are you", "come stai"),
    ("i love machine learning", "amo il machine learning")
    ]

    with open("src_vocab.json","r") as f:
       src_vocab = json.load(f)

    with open("tgt_vocab.json","r") as f:
       tgt_vocab = json.load(f)

    dataset= Dataset(data_pair, "src_vocab.json", "tgt_vocab.json", src_vocab, tgt_vocab)

    print(dataset[1])
    


  
        

        





