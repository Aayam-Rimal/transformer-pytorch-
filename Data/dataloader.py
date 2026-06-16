from torch.utils.data import DataLoader
from .dataset import Dataset
import torch
import json

def pad_sequence(seq,max_len,pad_value=0):

    return seq + [pad_value] * (max_len- len(seq))

def collate(batch, max_len=512):

    src,tgt_inp,tgt_out= zip(*batch)

    src = [x[:max_len] for x in src]
    tgt_inp = [x[:max_len] for x in tgt_inp]
    tgt_out = [x[:max_len] for x in tgt_out]

    src_max= max(len(x) for x in src )
    tgt_max= max(len(x) for x in tgt_inp)

    src= [pad_sequence(x,src_max) for x in src]
    tgt_inp= [pad_sequence(x,tgt_max) for x in tgt_inp]
    tgt_out = [pad_sequence(x,tgt_max) for x in tgt_out]

    return {
        "src": torch.tensor(src),
        "tgt_in": torch.tensor(tgt_inp),
        "tgt_out": torch.tensor(tgt_out)
    }

data_pair = [
    ("hello world", "ciao mondo"),
    ("how are you", "come stai"),
    ("i love machine learning", "amo il machine learning")
]

with open("src_vocab.json","r") as f:
       src_vocab = json.load(f)

with open("tgt_vocab.json","r") as f:
       tgt_vocab = json.load(f)


dataset= Dataset(data_pair, src_vocab_path="src_vocab.json", tgt_vocab_path="tgt_vocab.json",src_vocab=src_vocab,tgt_vocab=tgt_vocab )


train_loader= DataLoader(
    dataset,
    batch_size=3,
    shuffle=True,
    collate_fn=collate
)

print(next(iter(train_loader)))








