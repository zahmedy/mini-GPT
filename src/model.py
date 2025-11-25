

import torch 
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    BLOCK_SIZE,
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    FFN_DIM,
    DROPOUT,
    DEVICE
)


class CausalSelfAttention(nn.Module):
    """
    A single multi-head causal self-attention layer.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Causal mask: upper triangular (T, T)
        # buffer = not a parameter, but moved to correct device with the module
        mask = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        """
        x: (B, T, E)
        returns: (B, T, E)
        """
        B, T, E = x.shape

        # 1. Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into heads: (B, T, E) -> (B, num_heads, T, head_dim)
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        # shapes: (B, H, T, D)

        # Scaled dot-product attention
        # scores: (B, H, T, T)
        scores = q @ k.transpose(-2, -1) / (self.head_dim**0.5)

        # Apply causal mask: prevent attending to future positions
        # causal_mask: (T, T) with 1s in lower triangle
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # softmask over "key" dimension
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values 
        out = attn @ v # (B, H, T, D)

        # Merge heads: (B, H, T, D) -> (B, T, E)
        out = out.transpose(1, 2).contiguous().view(B, T, E)

        # Final linear projection 
        out = self.out_proj(out)
        return out
    

class TransformerBlock(nn.Module):
    """
    One Transformer block: (x + self-attn) + feed-forward, with LayerNorm and residuals.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, T, E)

        # self-attention with residual connection 
        x = x + self.attn(self.ln1(x))

        # feed-forward with residual connection 
        x = x + self.ffn(self.ln2(x))

        return x  # (B, T, E)
    

class MiniGPT(nn.Module):
    """
    Tiny GPT-style decoder-only Transformer for character-level language modeling.
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

        # Token & positional embeddings
        self.token_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)

        # Stack of transformer blocks 
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=EMBED_DIM,
                    num_heads=NUM_HEADS,
                    ffn_dim=FFN_DIM,
                    dropout=DROPOUT,
                )
                for _ in range(NUM_LAYERS)
            ]
        )

        self.ln_final = nn.LayerNorm(EMBED_DIM)

        # Final projection to vocabulary logits
        self.head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, idx):
        """
        idx: (B, T) of token indices
        returns: logits of shape (B, T, vocab_size)
        """
        B, T = idx.shape

        # Toke embeddings
        tok_emb = self.token_emb(idx)   # (B, T, E)

        # Positional embeddings: positions 0..T-1
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(positions)[None, :, :] # (1, T, E)

        # Combine token + positional info
        x = tok_emb + pos_emb  # (B, T, E)

        # pass through Transformer blocks 
        for block in self.blocks:
            x = block(x)

        # final layer norm
        x = self.ln_final(x)    # (B, T, E)

        # Output logits over vocabulary
        logits = self.head(x)   # (B, T, vocab_size)

        return logits
    

if __name__ == "__main__":
    # Simple sanity check with dummy vocab size and random input
    vocab_size = 100
    model = MiniGPT(vocab_size).to(DEVICE)

    B, T = 4, 16
    dummy_idx = torch.randint(0, vocab_size, (B, T), device=DEVICE)
    logits = model(dummy_idx)
    print("Logits shape:", logits.shape)  # expected: (4, 16, 100)
