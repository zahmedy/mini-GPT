# Mini-GPT

A simple GPT implementation built for learning how transformers work. This code prioritizes clarity over optimization - every operation is explicit and readable with print statements showing tensor shapes and attention patterns at each step.

## What is This?

This is a from-scratch transformer implementation designed for understanding how these models work:

- Clarity over optimization - every operation is explicit
- Extensive visualization - prints show tensor shapes and attention patterns
- Minimal dependencies - just PyTorch
- Educational comments throughout

You'll learn about tokenization, embeddings, multi-head attention, causal masking, feed-forward networks, layer normalization, residual connections, and autoregressive generation.

## Architecture

```
Input Text: "hello world"
      ↓
[Tokenization] → [104, 101, 108, 108, 111, ...]
      ↓
[Token Embeddings] + [Positional Embeddings]
      ↓
┌─────────────────────────────────┐
│   Transformer Block 1           │
│  ┌──────────────────────────┐   │
│  │  Layer Norm              │   │
│  │         ↓                │   │
│  │  Multi-Head Attention    │   │  ← Q, K, V projections
│  │    • Split into heads    │   │  ← Scaled dot-product
│  │    • Causal masking      │   │  ← Softmax weights
│  │    • Attention @ Values  │   │
│  └──────────────────────────┘   │
│         ↓                        │
│  [Residual Connection]           │
│         ↓                        │
│  ┌──────────────────────────┐   │
│  │  Layer Norm              │   │
│  │         ↓                │   │
│  │  Feed-Forward Network    │   │  ← Linear → ReLU → Linear
│  └──────────────────────────┘   │
│         ↓                        │
│  [Residual Connection]           │
└─────────────────────────────────┘
      ↓
   [Repeat for N layers]
      ↓
[Final Layer Norm]
      ↓
[Output Head] → Logits over vocabulary
      ↓
[Softmax] → Probabilities
      ↓
[Sample] → Next token prediction
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python -m src.train

# Generate text
python -m src.generate
```

The training script shows dataset creation, model architecture details, attention mechanism breakdown, training progress, and gradient statistics. The generation script shows step-by-step token generation with probability distributions.

## Project Structure

```
mini-GPT/
├── data/
│   └── tiny_corpus.txt
├── src/
│   ├── config.py         # Hyperparameters
│   ├── tokenizer.py      # Character-level tokenization
│   ├── data.py           # Dataset and DataLoader
│   ├── model.py          # Transformer implementation
│   ├── train.py          # Training loop
│   └── generate.py       # Text generation
└── saved_models/
    └── mini_gpt.pth
```

## Understanding the Code

### Tokenization

Character-level tokenizer that converts text to numbers:

```python
tokenizer = CharTokenizer(text)
tokens = tokenizer.encode("hello")  # [104, 101, 108, 108, 111]
text = tokenizer.decode(tokens)     # "hello"
```

### Embeddings

Each token gets token embeddings (what it means) and positional embeddings (where it is):

```python
tok_emb = self.token_emb(idx)
pos_emb = self.pos_emb(positions)
x = tok_emb + pos_emb
```

### Multi-Head Attention

The core mechanism - each token attends to previous tokens:

```python
# Project to Q, K, V
q = self.q_proj(x)
k = self.k_proj(x)
v = self.v_proj(x)

# Split into multiple heads
q = split_heads(q)

# Compute attention scores
scores = (q @ k.transpose(-2, -1)) / sqrt(head_dim)

# Apply causal mask
scores = scores.masked_fill(mask == 0, -inf)

# Get attention weights
attn = softmax(scores, dim=-1)

# Weighted sum of values
out = attn @ v
```

### Feed-Forward Network

Simple MLP after attention:

```python
ffn = nn.Sequential(
    nn.Linear(embed_dim, ffn_dim),
    nn.ReLU(),
    nn.Linear(ffn_dim, embed_dim),
    nn.Dropout(dropout)
)
```

### Text Generation

Generate text one token at a time:

```python
for step in range(max_new_tokens):
    logits = model(context)
    logits_last = logits[:, -1, :]
    logits_last = logits_last / temperature
    probs = softmax(logits_last)
    next_token = sample(probs)
    context = cat([context, next_token])
```

## Configuration

```python
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FFN_DIM = 512
BLOCK_SIZE = 32

BATCH_SIZE = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50
DROPOUT = 0.1
```

Automatically uses the best available device (CUDA, MPS, or CPU).

## Resources

Papers:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Tutorials:
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy

Videos:
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy

## License

MIT License
