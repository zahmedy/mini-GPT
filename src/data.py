from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import CharTokenizer
from .config import DATA_PATH, BLOCK_SIZE, BATCH_SIZE, RANDOM_SEED


class TextDataset(Dataset):
    """
    Takes a long string and creates (input, target) pairs.
    Each sample:
        x: BLOCK_SIZE characters
        y: BLOCK_SIZE characters (x shifted by one position)
    """

    def __init__(self, text: str, tokenizer: CharTokenizer, block_size: int) -> None:
        self.text = text
        self.tokenizer = tokenizer
        self.block_size = block_size

        print(f"\n{'='*80}")
        print(f"CREATING DATASET")
        print(f"{'='*80}")
        print(f"Text length: {len(text)} characters")
        print(f"Block size: {block_size}")

        # Encode entire text into integers once
        encoded = tokenizer.encode(text)
        if len(encoded) <= block_size:
            raise ValueError(
                f"Text length ({len(encoded)}) must be greater than block_size ({block_size})."
            )
        self.data = torch.tensor(encoded, dtype=torch.long)
        print(f"Encoded data shape: {self.data.shape}")
        print(f"Number of training samples: {len(self.data) - block_size}")

    def __len__(self) -> int:
        # Number of possible blocks we can extract
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: characters from idx to idx+block_size
        x = self.data[idx: idx + self.block_size]
        # y: next characters, shifted by one
        y = self.data[idx + 1: idx + 1 + self.block_size]

        # Print first sample details
        if idx == 0:
            print(f"\n{'='*80}")
            print(f"FIRST SAMPLE (idx=0):")
            print(f"{'='*80}")
            print(f"Input  (x) shape: {x.shape}, values: {x[:10].tolist()}...")
            print(f"Target (y) shape: {y.shape}, values: {y[:10].tolist()}...")
            print(f"Input  text: '{self.tokenizer.decode(x.tolist())}'")
            print(f"Target text: '{self.tokenizer.decode(y.tolist())}'")

        return x, y
    

def load_text() -> str:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def get_dataloaders(val_ratio: float = 0.1) -> Tuple[DataLoader, DataLoader, CharTokenizer]:
    torch.manual_seed(RANDOM_SEED)

    print("\n" + "="*80)
    print("DATA LOADING")
    print("="*80)

    text = load_text()
    print(f"Loaded text with {len(text)} characters")
    print(f"First 100 chars: '{text[:100]}'")

    tokenizer = CharTokenizer(text)

    n = len(text)
    if n <= BLOCK_SIZE:
        raise ValueError(
            f"Corpus is too small (len={n}) for block_size={BLOCK_SIZE}. "
            "Add more text or reduce block_size."
        )

    # Ensure validation split has at least one block, but leave enough for training
    val_size = max(int(n * val_ratio), BLOCK_SIZE + 1)
    if n - val_size < BLOCK_SIZE + 1:
        val_size = BLOCK_SIZE + 1

    split_idx = n - val_size
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    print(f"\nData split:")
    print(f"  Train: {len(train_text)} chars")
    print(f"  Val:   {len(val_text)} chars")

    train_ds = TextDataset(train_text, tokenizer, BLOCK_SIZE)
    val_ds = TextDataset(val_text, tokenizer, BLOCK_SIZE)

    train_dataloader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"\nDataLoader batches:")
    print(f"  Train batches: {len(train_dataloader)}")
    print(f"  Val batches:   {len(val_dataloader)}")
    print(f"  Batch size:    {BATCH_SIZE}")

    return train_dataloader, val_dataloader, tokenizer


if __name__ == "__main__":
    train_loader, val_loader, tokenizer = get_dataloaders()
    x, y = next(iter(train_loader))
    print("Batch x shape:", x.shape)  # [BATCH_SIZE, BLOCK_SIZE]
    print("Batch y shape:", y.shape)
    print("Vocab size:", tokenizer.vocab_size)
