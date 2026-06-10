from datasets import load_dataset
import json
import re


def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Zà-ÿ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocab(save_src="src_vocab.json", save_tgt="tgt_vocab.json"):

    ds = load_dataset("Helsinki-NLP/opus-100", "en-it")
    
    src_vocab= set()
    tgt_vocab= set()
    
    for sample in ds["train"]:
        en= clean(sample["translation"]["en"])
        it= clean(sample["translation"]["it"])
    
        src_vocab.update(en.split())
        tgt_vocab.update(it.split())
    
    src_id= {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3
    }
    
    tgt_id= {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3
    }
    
    for word in sorted(src_vocab):
    
        if word not in src_id:
    
          src_id[word]= len(src_id)
    
    for word in sorted(tgt_vocab):
    
        if word not in tgt_id:
    
          tgt_id[word]= len(tgt_id)
    
    
    with open(save_src,"w") as f:
        json.dump(src_id,f)
    
    with open(save_tgt,"w") as f:
        json.dump(tgt_id,f)


build_vocab()