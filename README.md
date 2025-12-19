# 🤖 Mini-GPT: Learn Transformers from First Principles

<div align="center">

**A minimalist, educational implementation of GPT for understanding transformer architecture**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

*Built for learning • Extensively documented • Visualization-first*

</div>

---

## 📚 What is This?

This is a **from-scratch implementation** of a GPT-style transformer, designed specifically for **learning and understanding** how these models work. Unlike production implementations, this codebase prioritizes:

- ✨ **Clarity over optimization** - Every operation is explicit and readable
- 🔍 **Extensive visualization** - Print statements show tensor shapes, attention patterns, and mathematical operations at every step
- 📖 **Educational comments** - Learn what each component does and why it matters
- 🧪 **Minimal dependencies** - Just PyTorch and basic Python libraries

### What You'll Learn

- **Tokenization**: How text is converted to numbers
- **Embeddings**: Token and positional representations
- **Multi-Head Attention**: The core mechanism that makes transformers work
- **Scaled Dot-Product Attention**: Q, K, V matrices and how attention scores are computed
- **Causal Masking**: Preventing the model from "cheating" by looking at future tokens
- **Feed-Forward Networks**: Non-linear transformations in each layer
- **Layer Normalization & Residual Connections**: Stabilizing deep networks
- **Autoregressive Generation**: How models generate text one token at a time
- **Backpropagation**: Watching gradients flow and weights update

---

## 🏗️ Architecture

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

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mini-GPT.git
cd mini-GPT

# Install dependencies
pip install -r requirements.txt
```

### Training

Train the model on your text corpus:

```bash
python -m src.train
```

**What you'll see:**
- 📊 Dataset creation with train/val split
- 🏗️ Model architecture details (parameters, layers, dimensions)
- 🔄 Token embedding and positional encoding visualization
- 🧠 Attention mechanism breakdown with Q, K, V matrices
- 📈 Training progress with loss metrics
- ⚡ Backpropagation with gradient statistics
- 💾 Model checkpointing

### Text Generation

Generate text from trained model:

```bash
python -m src.generate
```

**What you'll see:**
- 🎨 Step-by-step token generation
- 📊 Probability distributions over vocabulary
- 🎲 Top-K sampling visualization
- 🌡️ Temperature effects on randomness
- 🔍 Attention patterns for each prediction

---

## 📂 Project Structure

```
mini-GPT/
│
├── README.md                      # You are here
├── requirements.txt               # Python dependencies
│
├── data/
│   └── tiny_corpus.txt           # Training text (replace with your own!)
│
├── src/
│   ├── config.py                 # Hyperparameters and device configuration
│   ├── tokenizer.py              # Character-level tokenization
│   ├── data.py                   # Dataset and DataLoader creation
│   ├── model.py                  # 🌟 Transformer implementation
│   ├── train.py                  # Training loop with visualization
│   └── generate.py               # Autoregressive text generation
│
└── saved_models/
    └── mini_gpt.pth              # Trained model checkpoint
```

---

## 🧠 Understanding the Code

### 1. Tokenization ([`src/tokenizer.py`](src/tokenizer.py))

Converts text to numerical tokens:

```python
# Character-level tokenizer
tokenizer = CharTokenizer(text)
tokens = tokenizer.encode("hello")  # → [104, 101, 108, 108, 111]
text = tokenizer.decode(tokens)     # → "hello"
```

**Visualization output:**
```
TOKENIZER INITIALIZATION
Total unique characters: 65
Vocabulary: ['<unk>', ' ', '!', ',', '.', 'a', 'b', 'c', ...]
Vocab size: 65
```

### 2. Embeddings ([`src/model.py:280-297`](src/model.py#L280-L297))

Each token gets two types of embeddings:
- **Token embeddings**: What the token means
- **Positional embeddings**: Where the token is in the sequence

```python
tok_emb = self.token_emb(idx)      # (Batch, Time, Embed_dim)
pos_emb = self.pos_emb(positions)  # (1, Time, Embed_dim)
x = tok_emb + pos_emb              # Combined representation
```

### 3. Multi-Head Attention ([`src/model.py:48-144`](src/model.py#L48-L144))

The core of the transformer! Each token attends to previous tokens:

```python
# 1. Project input to Q, K, V
q = self.q_proj(x)  # Query: "What am I looking for?"
k = self.k_proj(x)  # Key: "What do I contain?"
v = self.v_proj(x)  # Value: "What should I output?"

# 2. Split into multiple attention heads
q = split_heads(q)  # (Batch, Heads, Time, Head_dim)

# 3. Compute attention scores
scores = (q @ k.transpose(-2, -1)) / sqrt(head_dim)

# 4. Apply causal mask (can't see future tokens)
scores = scores.masked_fill(mask == 0, -inf)

# 5. Softmax to get attention weights
attn = softmax(scores, dim=-1)

# 6. Weighted sum of values
out = attn @ v
```

**Visualization output:**
```
🔍 ATTENTION LAYER
  1️⃣  LINEAR PROJECTIONS (Q, K, V)
     Q shape: (4, 32, 128)

  2️⃣  SPLIT INTO 4 HEADS
     Each head processes 32 dimensions

  3️⃣  SCALED DOT-PRODUCT ATTENTION
     Q @ K^T shape: (4, 4, 32, 32)
     Scaling factor: 0.1768

  4️⃣  CAUSAL MASKING
     Example mask (5x5):
     [[1, 0, 0, 0, 0],
      [1, 1, 0, 0, 0],
      [1, 1, 1, 0, 0],
      [1, 1, 1, 1, 0],
      [1, 1, 1, 1, 1]]

  5️⃣  SOFTMAX → ATTENTION WEIGHTS
     Sample pattern: [1.0, 0.0, 0.0, ...]
```

### 4. Feed-Forward Network ([`src/model.py:195-200`](src/model.py#L195-L200))

Simple MLP after attention:

```python
ffn = nn.Sequential(
    nn.Linear(embed_dim, ffn_dim),    # Expand
    nn.ReLU(),                         # Non-linearity
    nn.Linear(ffn_dim, embed_dim),    # Compress back
    nn.Dropout(dropout)
)
```

### 5. Autoregressive Generation ([`src/generate.py:88-161`](src/generate.py#L88-L161))

Generate text one token at a time:

```python
for step in range(max_new_tokens):
    # 1. Get model predictions
    logits = model(context)

    # 2. Take logits for last position
    logits_last = logits[:, -1, :]

    # 3. Apply temperature
    logits_last = logits_last / temperature

    # 4. Convert to probabilities
    probs = softmax(logits_last)

    # 5. Sample next token
    next_token = sample(probs)

    # 6. Append to context
    context = cat([context, next_token])
```

**Visualization output:**
```
Step 1/100
─────────────────────────────────────
🎲 Softmax → Probabilities:
   Top 5 predictions:
      1. 'e' (token 15) : 0.2456
      2. 't' (token 30) : 0.1832
      3. ' ' (token 1)  : 0.1245
      4. 'a' (token 11) : 0.0892
      5. 'o' (token 25) : 0.0654

✅ Sampled token: 'e' (ID: 15)
```

---

## ⚙️ Configuration ([`src/config.py`](src/config.py))

Customize the model architecture:

```python
# Model architecture
EMBED_DIM = 128        # Embedding dimension
NUM_HEADS = 4          # Number of attention heads
NUM_LAYERS = 4         # Number of transformer blocks
FFN_DIM = 512          # Feed-forward hidden dimension
BLOCK_SIZE = 32        # Context length (sequence length)

# Training
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50
DROPOUT = 0.1
```

### Hardware Support

Automatically uses the best available device:

```python
DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
```

---

## 📊 Training Output Example

```
################################################################################
#                         MINI-GPT TRAINING                                    #
################################################################################

DATA LOADING
================================================================================
Loaded text with 50000 characters
Vocab size: 65

🏗️  BUILDING MiniGPT MODEL
================================================================================
Embed dim: 128
Num heads: 4
Num layers: 4
Total parameters: 234,817

⚙️  TRAINING CONFIGURATION
================================================================================
Learning rate: 0.0003
Optimizer: Adam
Loss function: CrossEntropyLoss

################################################################################
# EPOCH 1/50
################################################################################

🏋️  TRAINING EPOCH
📦 BATCH 1/156
   Input (x) shape: (8, 32)
   Target (y) shape: (8, 32)

🚀 MODEL FORWARD PASS
1️⃣  TOKEN EMBEDDINGS
   Embedding lookup: 65 vocab → 128 dimensions

2️⃣  POSITIONAL EMBEDDINGS
   Positions: [0, 1, 2, 3, 4, ...]

3️⃣  COMBINED EMBEDDINGS
   Combined shape: (8, 32, 128)

🧱 TRANSFORMER BLOCK #1
  🔍 ATTENTION LAYER
    Q @ K^T shape: (8, 4, 32, 32)
    Attention weights sum to 1.0

  🔄 FEED-FORWARD NETWORK
    FFN: Linear(128 → 512) → ReLU → Linear(512 → 128)

🎯 LOSS CALCULATION
   Cross Entropy Loss: 4.1743

⚡ BACKPROPAGATION
   Sample gradient statistics:
      Param 0: grad mean=-0.000123, grad norm=0.456789
   ✅ Weights updated!

📈 EPOCH 1 SUMMARY
Train Loss: 3.8945
Val Loss: 3.7234
✅ Saved new best model! (val loss = 3.7234)
```

---

## 🎨 Generation Output Example

```
🎨 TEXT GENERATION (Autoregressive)
Prompt: 'the'
Max new tokens: 100
Temperature: 1.0

Step 1/100
─────────────────────────────────────
📊 Model output:
   Logits shape: (1, 3, 65)
   Using last position logits: (1, 65)

🎲 Softmax → Probabilities:
   Top 5 predictions:
      1. ' ' (token 1)  : 0.3421
      2. 'r' (token 28) : 0.1234
      3. 'n' (token 24) : 0.0987
      4. 'y' (token 35) : 0.0765
      5. 'm' (token 23) : 0.0543

✅ Sampled token: ' ' (ID: 1)

the sun was shining on the sea, shining with all its might:
he did his very best to make the billows smooth and bright--
```

---

## 🎓 Learning Path

### For Beginners

1. **Start with [`src/tokenizer.py`](src/tokenizer.py)**
   - Understand how text becomes numbers
   - Run it standalone to see encoding/decoding

2. **Explore [`src/data.py`](src/data.py)**
   - See how training data is created
   - Understand input/target pairs

3. **Read [`src/model.py`](src/model.py) top-to-bottom**
   - `MiniGPT` class: The full model
   - `TransformerBlock`: One layer
   - `CausalSelfAttention`: The attention mechanism

4. **Run training and watch the output**
   - See tensor shapes flow through the network
   - Understand what each operation does

### For Intermediate Learners

- Modify hyperparameters in [`src/config.py`](src/config.py) and observe effects
- Experiment with different attention patterns
- Try different sampling strategies (temperature, top-k, top-p)
- Visualize attention weights
- Compare model performance with different architectures

### For Advanced Learners

- Implement improvements:
  - Flash Attention
  - Rotary Position Embeddings (RoPE)
  - Group Query Attention (GQA)
  - Layer normalization variants (RMSNorm)
- Add features:
  - Mixed precision training
  - Gradient accumulation
  - Learning rate scheduling
  - Model quantization

---

## 🔬 Key Features for Learning

### Extensive Visualization

Every component prints:
- **Tensor shapes** at each step
- **Mathematical operations** being performed
- **Attention patterns** and scores
- **Gradient statistics** during training
- **Probability distributions** during generation

### Clean, Readable Code

```python
# Before (typical production code)
x = self.attn(self.ln1(x)) + x

# After (educational code)
x_norm = self.ln1(x)           # Normalize first
attn_out = self.attn(x_norm)   # Apply attention
x = x + attn_out               # Residual connection
```

### No Hidden Abstractions

- No external transformer libraries
- Every operation is explicit
- Full control over the architecture

---

## 📈 Performance Tips

### For Training

1. **Increase batch size** if you have more GPU memory
2. **Use larger BLOCK_SIZE** for better context
3. **More layers/heads** for more capacity (but slower)
4. **Gradient accumulation** for larger effective batch size

### For Generation

1. **Lower temperature** (0.7-1.0) for more coherent text
2. **Higher temperature** (1.2-2.0) for more creative text
3. **Adjust top-k** to control diversity
4. **Longer prompts** give more context

---

## 🤝 Contributing

This is an educational project! Contributions that improve:
- **Documentation and explanations**
- **Visualization and debugging tools**
- **Code clarity** (not optimization)
- **Educational notebooks**

are highly welcome!

---

## 📚 Resources to Learn More

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper

### Tutorials
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) by Lilian Weng
- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy

### Videos
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy
- [Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc) by 3Blue1Brown

---

## 📝 License

MIT License - Feel free to use this for learning, teaching, or building upon!

---

## 🙏 Acknowledgments

- Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and [minGPT](https://github.com/karpathy/minGPT)
- Built for educational purposes with visualization in mind
- Thanks to the PyTorch team for making deep learning accessible

---

## 💬 Questions?

This project is meant for learning! If you have questions about:
- How a specific component works
- Why certain design decisions were made
- How to extend or modify the code

Feel free to open an issue for discussion.

---

<div align="center">

**⭐ If this helped you understand transformers, please star the repo! ⭐**

Made with ❤️ for learners by learners

</div>
