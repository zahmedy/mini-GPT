# mini-GPT
Simple text generation model

```mini_gpt/
│
├─ README.md
├─ requirements.txt
│
├─ data/
│   └─ tiny_corpus.txt        # our training text
│
├─ notebooks/
│   └─ 01_explore_tokens.ipynb  # explore tokenization, embeddings
│
├─ src/
│   ├─ config.py              # hyperparameters
│   ├─ data.py                # load text, build dataset
│   ├─ tokenizer.py           # simple tokenizer (char-level or word-level)
│   ├─ model.py               # tiny GPT-style Transformer
│   ├─ train.py               # training loop
│   ├─ generate.py            # text sampling script
│   └─ utils.py               # helper functions
│
└─ saved_models/
    └─ mini_gpt.pth
```
