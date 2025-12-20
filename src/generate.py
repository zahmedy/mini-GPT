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
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Train first with `python3 -m src.train`."
        )

    print(f"Loading model from: {checkpoint_path}")

    # Rebuild tokenizer from full corpus text
    text = load_text()
    tokenizer = CharTokenizer(text)

    # Load checkpoint (weights + vocab size)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    vocab_size = checkpoint["vocab_size"]

    model = MiniGPT(vocab_size).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded successfully on {DEVICE}")

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

    # Encode the prompt to token ids
    start_ids = tokenizer.encode(start_text)
    idx = torch.tensor([start_ids], dtype=torch.long, device=DEVICE)

    print(f"Generating from prompt: '{start_text}'")

    for step in range(max_new_tokens):
        # If sequence is longer than BLOCK_SIZE, keep only the last BLOCK_SIZE
        if idx.size(1) > BLOCK_SIZE:
            idx_cond = idx[:, -BLOCK_SIZE:]
        else:
            idx_cond = idx

        # Forward pass: get logits for each position
        logits = model(idx_cond)

        # Take logits for the LAST time step
        logits_last = logits[:, -1, :]

        # Adjust temperature
        logits_last = logits_last / temperature

        # Convert to probabilities
        probs = F.softmax(logits_last, dim=-1)

        # Top-K sampling
        def top_k_filter(probs: torch.Tensor, k: int = 20) -> torch.Tensor:
            k = min(k, probs.size(-1))
            values, indices = torch.topk(probs, k)
            mask = torch.zeros_like(probs)
            mask.scatter_(1, indices, values)
            filtered_probs = mask / (mask.sum(dim=-1, keepdim=True) + 1e-8)
            return filtered_probs

        probs = top_k_filter(probs, k=30)

        # Sample next token id
        next_id = torch.multinomial(probs, num_samples=1)
        next_char = tokenizer.decode([next_id.item()])

        # Append to the sequence
        idx = torch.cat([idx, next_id], dim=1)

        # Print the character as it's generated
        print(next_char, end='', flush=True)

    # Decode the full sequence (including the prompt)
    out_ids = idx[0].tolist()
    result = tokenizer.decode(out_ids)

    print(f"\n\nGeneration complete ({len(out_ids)} tokens)")

    return result


def main() -> None:
    print("MINI-GPT TEXT GENERATION")
    print("="*80)

    model, tokenizer = load_model_and_tokenizer()

    while True:
        prompt = input("\nEnter a prompt (or empty to quit): ")
        prompt = prompt.lower()
        if prompt.strip() == "":
            print("Goodbye!")
            break

        generated = generate(
            model,
            tokenizer,
            start_text=prompt,
            max_new_tokens=100,
            temperature=1.5,
        )
        print(f"\n\nFull output:\n{generated}")


if __name__ == "__main__":
    main()
