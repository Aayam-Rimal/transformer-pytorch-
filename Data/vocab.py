from datasets import load_dataset
import json

ds = load_dataset("Helsinki-NLP/opus-100", "en-it")

src_vocab= set()
tgt_vocab= set()

for sample in ds["train"]:
    en= sample["translation"]["en"]
    it= sample["translation"]["it"]

    en_tokens= en.lower().split()
    it_tokens= it.lower().split()

    src_vocab.update(en_tokens)
    tgt_vocab.update(it_tokens)

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


with open("src_vocab.json","w") as f:
    json.dump(src_id,f)

with open("tgt_vocab.json","w") as f:
    json.dump(tgt_id,f)