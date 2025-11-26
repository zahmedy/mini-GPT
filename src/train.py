# src/train.py
import os

import torch
import torch.nn as nn
import torch.optim as optim

from .data import get_dataloaders
from .model import MiniGPT
from .config import DEVICE, LEARNING_RATE, NUM_EPOCHS


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    total_tokens = 0

    for x, y in dataloader:
        # x, y: (B, T)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # 1. Forward pass: logits over vocab for each position
        logits = model(x)  # (B, T, vocab_size)

        # 2. Reshape for CrossEntropyLoss:
        #    expects (N, C) and (N,)
        B, T, V = logits.shape
        logits = logits.view(B * T, V)
        y = y.view(B * T)

        loss = criterion(logits, y)

        # 3. Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * B * T
        total_tokens += B * T

    avg_loss = running_loss / total_tokens
    return avg_loss


def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
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


def main():
    os.makedirs("saved_models", exist_ok=True)

    # 1. Get data + tokenizer
    train_loader, val_loader, tokenizer = get_dataloaders()

    vocab_size = tokenizer.vocab_size
    print("Vocab size:", vocab_size)

    # 2. Create model
    model = MiniGPT(vocab_size).to(DEVICE)
    print("Using device:", DEVICE)

    # 3. Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    # 4. Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                },
                "saved_models/mini_gpt.pth",
            )
            print(f"✅ Saved new best model (val loss = {best_val_loss:.4f})")


if __name__ == "__main__":
    main()
