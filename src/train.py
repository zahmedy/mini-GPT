# src/train.py
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .data import get_dataloaders
from .model import MiniGPT
from .config import DEVICE, LEARNING_RATE, NUM_EPOCHS


def train_one_epoch(
    model: MiniGPT,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> float:
    model.train()
    running_loss = 0.0
    total_tokens = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        B, T, V = logits.shape

        logits = logits.view(B * T, V)
        y = y.view(B * T)

        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * B * T
        total_tokens += B * T

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg_loss_so_far = running_loss / total_tokens
            print(f"   Batch {batch_idx + 1}/{len(dataloader)}: avg_loss={avg_loss_so_far:.4f}")

    avg_loss = running_loss / total_tokens
    return avg_loss


def evaluate(
    model: MiniGPT,
    dataloader: DataLoader,
    criterion: nn.Module,
) -> float:
    model.eval()
    running_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            y = y.view(B * T)

            loss = criterion(logits, y)

            running_loss += loss.item() * B * T
            total_tokens += B * T

    avg_loss = running_loss / total_tokens
    return avg_loss


def main() -> None:
    print("MINI-GPT TRAINING")
    print("="*80)

    os.makedirs("saved_models", exist_ok=True)

    train_loader, val_loader, tokenizer = get_dataloaders()
    vocab_size = tokenizer.vocab_size

    model = MiniGPT(vocab_size).to(DEVICE)
    print(f"Device: {DEVICE}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Learning rate: {LEARNING_RATE}, Epochs: {NUM_EPOCHS}")
    print("="*80)

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEPOCH {epoch}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                },
                "saved_models/mini_gpt.pth",
            )
            print(f"✅ New best model saved! (val loss = {best_val_loss:.4f})")
        else:
            print(f"Best val loss: {best_val_loss:.4f}")

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print("Model saved to: saved_models/mini_gpt.pth")


if __name__ == "__main__":
    main()
