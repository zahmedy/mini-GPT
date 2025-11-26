# src/generate.py

import torch
import torch.nn.functional as F

from .model import MiniGPT
from .data import load_text
from .tokenizer import CharTokenizer
from .config import DEVICE, BLOCK_SIZE


def load_model_and_tokenizer(checkpoint_path: str = "saved_models/mini_gpt.pth"):
    """
    Load trained MiniGPT model + tokenizer built from the same corpus.
    """
    # 1. Rebuild tokenizer from full corpus text
    text = load_text()
    tokenizer = CharTokenizer(text)

    # 2. Load checkpoint (weights + vocab size)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    vocab_size = checkpoint["vocab_size"]

    model = MiniGPT(vocab_size).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    start_text: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
):
    """
    Autoregressive text generation:
    - start from `start_text`
    - repeatedly predict next char and append it
    """
    model.eval()

    # 1. Encode the prompt to token ids
    start_ids = tokenizer.encode(start_text)
    # shape: (1, T)
    idx = torch.tensor([start_ids], dtype=torch.long, device=DEVICE)

    for _ in range(max_new_tokens):
        # 2. If sequence is longer than BLOCK_SIZE, keep only the last BLOCK_SIZE
        if idx.size(1) > BLOCK_SIZE:
            idx_cond = idx[:, -BLOCK_SIZE:]
        else:
            idx_cond = idx

        # 3. Forward pass: get logits for each position
        logits = model(idx_cond)  # (1, T_cond, vocab_size)

        # 4. Take logits for the LAST time step
        logits_last = logits[:, -1, :]  # (1, vocab_size)

        # 5. Optionally adjust temperature (controls randomness)
        logits_last = logits_last / temperature

        # 6. Convert to probabilities
        probs = F.softmax(logits_last, dim=-1)  # (1, vocab_size)

        # 7. Sample next token id
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # 8. Append to the sequence
        idx = torch.cat([idx, next_id], dim=1)  # (1, T+1)

    # 9. Decode the full sequence (including the prompt)
    out_ids = idx[0].tolist()
    return tokenizer.decode(out_ids)


def main():
    model, tokenizer = load_model_and_tokenizer()

    # Try a few different prompts
    while True:
        prompt = input("\nEnter a prompt (or empty to quit): ")
        if prompt.strip() == "":
            break

        print("\nGenerating...\n")
        generated = generate(
            model,
            tokenizer,
            start_text=prompt,
            max_new_tokens=20,
            temperature=0.7,  # try 0.7–1.2
        )
        print(generated)


if __name__ == "__main__":
    main()
