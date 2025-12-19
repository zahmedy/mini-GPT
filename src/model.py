

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
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.call_count = 0  # Track how many times forward is called

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, E)
        returns: (B, T, E)
        """
        B, T, E = x.shape

        # Only print detailed info for first few calls
        verbose = self.call_count < 2
        if verbose:
            print(f"\n{'─'*80}")
            print(f"  🔍 ATTENTION LAYER (Call #{self.call_count + 1})")
            print(f"{'─'*80}")
            print(f"  Input shape: (B={B}, T={T}, E={E})")

        # 1. Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if verbose:
            print(f"\n  1️⃣  LINEAR PROJECTIONS (Q, K, V)")
            print(f"     Q shape after projection: {q.shape} (Batch, Time, Embed)")
            print(f"     K shape after projection: {k.shape}")
            print(f"     V shape after projection: {v.shape}")

        # Split into heads: (B, T, E) -> (B, num_heads, T, head_dim)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        # shapes: (B, H, T, D)

        if verbose:
            print(f"\n  2️⃣  SPLIT INTO {self.num_heads} HEADS")
            print(f"     Q shape: {q.shape} (Batch, Heads, Time, Head_dim)")
            print(f"     Each head processes {self.head_dim} dimensions")

        # Scaled dot-product attention
        # scores: (B, H, T, T)
        scores = q @ k.transpose(-2, -1) / (self.head_dim**0.5)

        if verbose:
            print(f"\n  3️⃣  SCALED DOT-PRODUCT ATTENTION")
            print(f"     Q @ K^T shape: {scores.shape} (Batch, Heads, Time, Time)")
            print(f"     Scaling factor (1/√d_k): {1.0 / (self.head_dim**0.5):.4f}")
            print(f"     Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")

        # Apply causal mask: prevent attending to future positions
        # causal_mask: (T, T) with 1s in lower triangle
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        if verbose:
            print(f"\n  4️⃣  CAUSAL MASKING")
            print(f"     Mask shape: {mask.shape}")
            print(f"     Masked positions (upper triangle) set to -inf")
            print(f"     Example mask (first 5x5):")
            print(f"     {mask[:5, :5].int()}")

        # softmask over "key" dimension
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        if verbose:
            print(f"\n  5️⃣  SOFTMAX → ATTENTION WEIGHTS")
            print(f"     Attention shape: {attn.shape}")
            print(f"     Attention weights sum to 1.0 across key dimension")
            print(f"     Sample attention pattern (head 0, position 0): {attn[0, 0, 0, :5].tolist()}")

        # Weighted sum of values
        out = attn @ v # (B, H, T, D)

        if verbose:
            print(f"\n  6️⃣  WEIGHTED SUM OF VALUES")
            print(f"     Output shape: {out.shape} (Batch, Heads, Time, Head_dim)")
            print(f"     This is: Attention @ V")

        # Merge heads: (B, H, T, D) -> (B, T, E)
        out = out.transpose(1, 2).contiguous().view(B, T, E)

        if verbose:
            print(f"\n  7️⃣  MERGE HEADS")
            print(f"     Output shape: {out.shape} (Batch, Time, Embed)")

        # Final linear projection
        out = self.out_proj(out)

        if verbose:
            print(f"\n  8️⃣  OUTPUT PROJECTION")
            print(f"     Final output shape: {out.shape}")
            print(f"     Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")

        self.call_count += 1
        return out
    

class TransformerBlock(nn.Module):
    """
    One Transformer block: (x + self-attn) + feed-forward, with LayerNorm and residuals.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()

        self.call_count = 0  # Track how many times forward is called

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, E)

        verbose = self.call_count < 1  # Only print for first block
        if verbose:
            print(f"\n{'='*80}")
            print(f"🧱 TRANSFORMER BLOCK #{self.call_count + 1}")
            print(f"{'='*80}")
            print(f"Input shape: {x.shape}")
            print(f"Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")

        # self-attention with residual connection
        x_norm = self.ln1(x)
        if verbose:
            print(f"\n📊 Layer Norm 1:")
            print(f"   After LayerNorm shape: {x_norm.shape}")
            print(f"   Mean: {x_norm.mean().item():.4f}, Std: {x_norm.std().item():.4f}")

        attn_out = self.attn(x_norm)
        if verbose:
            print(f"\n➕ Residual Connection 1: x = x + attention(LayerNorm(x))")
            print(f"   Attention output shape: {attn_out.shape}")

        x = x + attn_out

        # feed-forward with residual connection
        x_norm2 = self.ln2(x)
        if verbose:
            print(f"\n📊 Layer Norm 2:")
            print(f"   After LayerNorm shape: {x_norm2.shape}")

        ffn_out = self.ffn(x_norm2)
        if verbose:
            print(f"\n🔄 FEED-FORWARD NETWORK")
            print(f"   FFN structure: Linear({x_norm2.shape[-1]} → {self.ffn[0].out_features}) → ReLU → Linear({self.ffn[0].out_features} → {ffn_out.shape[-1]})")
            print(f"   FFN output shape: {ffn_out.shape}")
            print(f"   FFN output range: [{ffn_out.min().item():.4f}, {ffn_out.max().item():.4f}]")

            print(f"\n➕ Residual Connection 2: x = x + ffn(LayerNorm(x))")

        x = x + ffn_out

        if verbose:
            print(f"\n✅ Block output shape: {x.shape}")
            print(f"   Output range: [{x.min().item():.4f}, {x.max().item():.4f}]")

        self.call_count += 1
        return x  # (B, T, E)
    

class MiniGPT(nn.Module):
    """
    Tiny GPT-style decoder-only Transformer for character-level language modeling.
    """
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.forward_count = 0

        print(f"\n{'='*80}")
        print(f"🏗️  BUILDING MiniGPT MODEL")
        print(f"{'='*80}")
        print(f"Vocab size: {vocab_size}")
        print(f"Embed dim: {EMBED_DIM}")
        print(f"Num heads: {NUM_HEADS}")
        print(f"Num layers: {NUM_LAYERS}")
        print(f"FFN dim: {FFN_DIM}")
        print(f"Block size: {BLOCK_SIZE}")
        print(f"Dropout: {DROPOUT}")

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

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\nTotal parameters: {total_params:,}")
        print(f"{'='*80}")

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T) of token indices
        returns: logits of shape (B, T, vocab_size)
        """
        B, T = idx.shape

        verbose = self.forward_count < 1  # Only print for first forward pass
        if verbose:
            print(f"\n{'#'*80}")
            print(f"🚀 MODEL FORWARD PASS #{self.forward_count + 1}")
            print(f"{'#'*80}")
            print(f"Input token indices shape: (B={B}, T={T})")
            print(f"Sample tokens (first sequence): {idx[0, :10].tolist()}...")

        # Token embeddings
        tok_emb = self.token_emb(idx)   # (B, T, E)

        if verbose:
            print(f"\n1️⃣  TOKEN EMBEDDINGS")
            print(f"   Embedding lookup: {self.vocab_size} vocab → {EMBED_DIM} dimensions")
            print(f"   Token embeddings shape: {tok_emb.shape}")
            print(f"   Sample embedding (token 0, first 5 dims): {tok_emb[0, 0, :5].tolist()}")

        # Positional embeddings: positions 0..T-1
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(positions)[None, :, :]  # (1, T, E)

        if verbose:
            print(f"\n2️⃣  POSITIONAL EMBEDDINGS")
            print(f"   Positions: {positions[:10].tolist()}...")
            print(f"   Positional embeddings shape: {pos_emb.shape}")
            print(f"   Sample pos embedding (pos 0, first 5 dims): {pos_emb[0, 0, :5].tolist()}")

        # Combine token + positional info
        x = tok_emb + pos_emb  # (B, T, E)

        if verbose:
            print(f"\n3️⃣  COMBINED EMBEDDINGS (Token + Position)")
            print(f"   Combined shape: {x.shape}")
            print(f"   This gives each token context-aware representation")

        # pass through Transformer blocks
        if verbose:
            print(f"\n4️⃣  TRANSFORMER LAYERS (x{NUM_LAYERS})")
            print(f"   Processing through {NUM_LAYERS} transformer blocks...")

        for i, block in enumerate(self.blocks):
            x = block(x)
            if verbose and i == 0:
                print(f"   [After block {i+1}] shape: {x.shape}")

        # final layer norm
        x = self.ln_final(x)    # (B, T, E)

        if verbose:
            print(f"\n5️⃣  FINAL LAYER NORM")
            print(f"   Shape: {x.shape}")
            print(f"   Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")

        # Output logits over vocabulary
        logits = self.head(x)   # (B, T, vocab_size)

        if verbose:
            print(f"\n6️⃣  OUTPUT HEAD (Linear: {EMBED_DIM} → {self.vocab_size})")
            print(f"   Logits shape: {logits.shape} (Batch, Time, Vocab)")
            print(f"   Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            print(f"   Sample logits (pos 0, first 10 vocab): {logits[0, 0, :10].tolist()}")
            print(f"\n{'#'*80}")
            print(f"✅ FORWARD PASS COMPLETE")
            print(f"{'#'*80}\n")

        self.forward_count += 1
        return logits
    

if __name__ == "__main__":
    # Simple sanity check with dummy vocab size and random input
    vocab_size = 100
    model = MiniGPT(vocab_size).to(DEVICE)

    B, T = 4, 16
    dummy_idx = torch.randint(0, vocab_size, (B, T), device=DEVICE)
    logits = model(dummy_idx)
    print("Logits shape:", logits.shape)  # expected: (4, 16, 100)
