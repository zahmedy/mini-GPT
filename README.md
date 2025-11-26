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

## Quickstart

1) Install deps (CPU wheels):
```bash
python3 -m pip install -r requirements.txt
```
2) Train:
```bash
python3 -m src.train
```
This uses `data/tiny_corpus.txt`, saves the best checkpoint to `saved_models/mini_gpt.pth`, and prints loss per epoch.

3) Generate text from a prompt:
```bash
python3 -m src.generate
```

## Notes
- `src/config.py` holds device selection (CUDA → MPS → CPU), context length (`BLOCK_SIZE`), and model hyperparameters tuned for the tiny sample corpus.
- `src/data.py` will raise a clear error if your corpus is shorter than `BLOCK_SIZE + 1`; lower the block size or add more text.
- `src/tokenizer.py` is a minimal character tokenizer. Replace with a wordpiece/BPE tokenizer if you need more expressive outputs.
