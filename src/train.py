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

    print(f"\n{'='*80}")
    print(f"🏋️  TRAINING EPOCH")
    print(f"{'='*80}")

    for batch_idx, (x, y) in enumerate(dataloader):
        # x, y: (B, T)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        verbose = batch_idx == 0  # Print details for first batch only

        if verbose:
            print(f"\n📦 BATCH {batch_idx + 1}/{len(dataloader)}")
            print(f"   Input (x) shape: {x.shape} (Batch, Sequence_Length)")
            print(f"   Target (y) shape: {y.shape}")
            print(f"   Sample input tokens:  {x[0, :10].tolist()}")
            print(f"   Sample target tokens: {y[0, :10].tolist()}")

        # 1. Forward pass: logits over vocab for each position
        logits = model(x)  # (B, T, vocab_size)

        # 2. Reshape for CrossEntropyLoss:
        #    expects (N, C) and (N,)
        B, T, V = logits.shape

        if verbose:
            print(f"\n🎯 LOSS CALCULATION")
            print(f"   Logits shape: {logits.shape} (Batch, Time, Vocab)")
            print(f"   Reshaping for CrossEntropyLoss:")
            print(f"     Logits: ({B}, {T}, {V}) → ({B * T}, {V})")
            print(f"     Targets: ({B}, {T}) → ({B * T},)")

        logits = logits.view(B * T, V)
        y = y.view(B * T)

        loss = criterion(logits, y)

        if verbose:
            print(f"\n   Cross Entropy Loss: {loss.item():.4f}")
            print(f"   (Lower is better - measures how well model predicts next token)")

        # 3. Backprop
        if verbose:
            print(f"\n⚡ BACKPROPAGATION")
            print(f"   1. Zero gradients")

        optimizer.zero_grad()

        if verbose:
            print(f"   2. Compute gradients (backward pass)")

        loss.backward()

        if verbose:
            # Show gradient statistics for a few parameters
            sample_params = list(model.parameters())[:3]
            print(f"   3. Sample gradient statistics:")
            for i, p in enumerate(sample_params):
                if p.grad is not None:
                    print(f"      Param {i}: grad mean={p.grad.mean().item():.6f}, "
                          f"grad norm={p.grad.norm().item():.6f}")

            print(f"   4. Update weights (optimizer.step)")

        optimizer.step()

        if verbose:
            print(f"   ✅ Weights updated!")

        running_loss += loss.item() * B * T
        total_tokens += B * T

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg_loss_so_far = running_loss / total_tokens
            print(f"   Batch {batch_idx + 1}/{len(dataloader)}: avg_loss={avg_loss_so_far:.4f}")

    avg_loss = running_loss / total_tokens
    print(f"\n✅ Epoch complete. Average loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(
    model: MiniGPT,
    dataloader: DataLoader,
    criterion: nn.Module,
) -> float:
    model.eval()
    running_loss = 0.0
    total_tokens = 0

    print(f"\n{'='*80}")
    print(f"📊 VALIDATION")
    print(f"{'='*80}")

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
    print(f"Validation complete. Average loss: {avg_loss:.4f}")
    return avg_loss


def main() -> None:
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*25 + "MINI-GPT TRAINING" + " "*36 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)

    os.makedirs("saved_models", exist_ok=True)

    # 1. Get data + tokenizer
    train_loader, val_loader, tokenizer = get_dataloaders()

    vocab_size = tokenizer.vocab_size

    # 2. Create model
    model = MiniGPT(vocab_size).to(DEVICE)
    print(f"\n💻 Using device: {DEVICE}")

    # 3. Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n{'='*80}")
    print(f"⚙️  TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Optimizer: Adam")
    print(f"Loss function: CrossEntropyLoss")
    print(f"{'='*80}")

    best_val_loss = float("inf")

    # 4. Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n\n{'#'*80}")
        print(f"# EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'#'*80}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)

        print(f"\n{'='*80}")
        print(f"📈 EPOCH {epoch} SUMMARY")
        print(f"{'='*80}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss if best_val_loss != float("inf") else 0
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                },
                "saved_models/mini_gpt.pth",
            )
            print(f"✅ Saved new best model! (val loss = {best_val_loss:.4f}, "
                  f"improved by {improvement:.4f})")
        else:
            print(f"   Best val loss so far: {best_val_loss:.4f}")

    print(f"\n\n{'#'*80}")
    print(f"# 🎉 TRAINING COMPLETE!")
    print(f"{'#'*80}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: saved_models/mini_gpt.pth")


if __name__ == "__main__":
    main()
