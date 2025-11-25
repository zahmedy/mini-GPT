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

    def __init__(self, text: str, tokenizer: CharTokenizer, block_size: int):
        self.text = text
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Encode entire text into integers once
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        # Number of possible blocks we can extract
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # x: characters from idx to idx+block_size
        x = self.data[idx: idx + self.block_size]
        # y: next characters, shifted by one
        y = self.data[idx + 1: idx + 1 + self.block_size]
        return x, y
    

def load_text():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def get_dataloaders(val_ratio: float = 0.1):
    torch.manual_seed(RANDOM_SEED)

    text = load_text()
    tokenizer =  CharTokenizer(text)

    # Split encoded data into train/val by character position
    n = len(text)
    split_idx = int(n * (1 - val_ratio))

    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_ds = TextDataset(train_text, tokenizer, BLOCK_SIZE)
    val_ds = TextDataset(val_text, tokenizer, BLOCK_SIZE)

    train_dataloader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer


if __name__ == "__main__":
    train_loader, val_loader, tokenizer = get_dataloaders()
    x, y = next(iter(train_loader))
    print("Batch x shape:", x.shape)  # [BATCH_SIZE, BLOCK_SIZE]
    print("Batch y shape:", y.shape)
    print("Vocab size:", tokenizer.vocab_size)