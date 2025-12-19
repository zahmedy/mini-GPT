# src/generate.py
import os
from typing import Tuple

import torch
import torch.nn.functional as F

from .model import MiniGPT
from .data import load_text
from .tokenizer import CharTokenizer
from .config import DEVICE, BLOCK_SIZE


def load_model_and_tokenizer(
    checkpoint_path: str = "saved_models/mini_gpt.pth",
) -> Tuple[MiniGPT, CharTokenizer]:
    """
    Load trained MiniGPT model + tokenizer built from the same corpus.
    """
    print("\n" + "="*80)
    print("🔄 LOADING MODEL")
    print("="*80)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Train first with `python3 -m src.train`."
        )

    print(f"Checkpoint path: {checkpoint_path}")

    # 1. Rebuild tokenizer from full corpus text
    text = load_text()
    tokenizer = CharTokenizer(text)

    # 2. Load checkpoint (weights + vocab size)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    vocab_size = checkpoint["vocab_size"]

    print(f"Loaded vocab size: {vocab_size}")

    model = MiniGPT(vocab_size).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Model loaded successfully!")
    print(f"Device: {DEVICE}")
    print("="*80)

    return model, tokenizer


@torch.no_grad()
def generate(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    start_text: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> str:
    """
    Autoregressive text generation:
    - start from `start_text`
    - repeatedly predict next char and append it
    """
    model.eval()

    print("\n" + "#"*80)
    print("🎨 TEXT GENERATION (Autoregressive)")
    print("#"*80)
    print(f"Prompt: '{start_text}'")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print("#"*80 + "\n")

    # 1. Encode the prompt to token ids
    start_ids = tokenizer.encode(start_text)
    # shape: (1, T)
    idx = torch.tensor([start_ids], dtype=torch.long, device=DEVICE)

    print(f"📝 Encoded prompt:")
    print(f"   Text: '{start_text}'")
    print(f"   Token IDs: {start_ids}")
    print(f"   Shape: {idx.shape}")

    print(f"\n🔄 Generating tokens one-by-one...\n")

    for step in range(max_new_tokens):
        # 2. If sequence is longer than BLOCK_SIZE, keep only the last BLOCK_SIZE
        if idx.size(1) > BLOCK_SIZE:
            idx_cond = idx[:, -BLOCK_SIZE:]
        else:
            idx_cond = idx

        verbose = step < 3  # Print details for first 3 tokens

        if verbose:
            print(f"{'─'*80}")
            print(f"Step {step + 1}/{max_new_tokens}")
            print(f"{'─'*80}")
            print(f"Current sequence length: {idx.size(1)}")
            print(f"Context (last 20 chars): '...{tokenizer.decode(idx[0].tolist())[-20:]}'")

        # 3. Forward pass: get logits for each position
        logits = model(idx_cond)  # (1, T_cond, vocab_size)

        # 4. Take logits for the LAST time step
        logits_last = logits[:, -1, :]  # (1, vocab_size)

        if verbose:
            print(f"\n📊 Model output:")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Using last position logits: {logits_last.shape}")
            print(f"   Logit range: [{logits_last.min().item():.3f}, {logits_last.max().item():.3f}]")

        # 5. Optionally adjust temperature (controls randomness)
        logits_last = logits_last / temperature

        if verbose:
            print(f"\n🌡️  Temperature scaling: {temperature}")
            print(f"   Scaled logits range: [{logits_last.min().item():.3f}, {logits_last.max().item():.3f}]")

        # 6. Convert to probabilities
        probs = F.softmax(logits_last, dim=-1)  # (1, vocab_size)

        if verbose:
            print(f"\n🎲 Softmax → Probabilities:")
            print(f"   Probs sum: {probs.sum().item():.6f} (should be ~1.0)")
            # Show top 5 predictions
            top_probs, top_indices = torch.topk(probs[0], 5)
            print(f"   Top 5 predictions (before top-k filtering):")
            for i, (prob, idx_pred) in enumerate(zip(top_probs, top_indices)):
                char = tokenizer.decode([idx_pred.item()])
                print(f"      {i+1}. '{char}' (token {idx_pred.item()}) : {prob.item():.4f}")

        # ---- TOP-K SAMPLING ----
        def top_k_filter(probs: torch.Tensor, k: int = 20) -> torch.Tensor:
            k = min(k, probs.size(-1))
            values, indices = torch.topk(probs, k)
            mask = torch.zeros_like(probs)
            mask.scatter_(1, indices, values)
            filtered_probs = mask / (mask.sum(dim=-1, keepdim=True) + 1e-8)
            return filtered_probs

        probs = top_k_filter(probs, k=30)   # you can tune k

        if verbose:
            print(f"\n🔝 Top-K filtering (k=30):")
            print(f"   Keeping only top 30 most likely tokens")

        # 7. Sample next token id
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        next_char = tokenizer.decode([next_id.item()])

        if verbose:
            print(f"\n✅ Sampled token: '{next_char}' (ID: {next_id.item()})")
            print(f"   Probability: {probs[0, next_id.item()].item():.4f}")

        # 8. Append to the sequence
        idx = torch.cat([idx, next_id], dim=1)  # (1, T+1)

        # Print the character as it's generated (non-verbose)
        if not verbose:
            print(next_char, end='', flush=True)

    # 9. Decode the full sequence (including the prompt)
    out_ids = idx[0].tolist()
    result = tokenizer.decode(out_ids)

    print("\n\n" + "="*80)
    print("✅ GENERATION COMPLETE")
    print("="*80)
    print(f"Total tokens generated: {len(out_ids)}")
    print(f"Final length: {len(result)} characters")

    return result


def main() -> None:
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*23 + "MINI-GPT TEXT GENERATION" + " "*32 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)

    model, tokenizer = load_model_and_tokenizer()

    # Try a few different prompts
    while True:
        prompt = input("\n\nEnter a prompt (or empty to quit): ")
        prompt = prompt.lower()
        if prompt.strip() == "":
            print("\n👋 Goodbye!")
            break

        generated = generate(
            model,
            tokenizer,
            start_text=prompt,
            max_new_tokens=100,
            temperature=1.5,  # try 0.7–1.2
        )
        print(f"\n\n📄 FULL OUTPUT:\n{generated}")


if __name__ == "__main__":
    main()
