"""
Small helper functions shared across the training and evaluation pipeline:
reproducibility, accuracy computation, plotting, and model checkpointing.
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """Fix random seeds so results are reproducible across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def accuracy(logits, labels):
    """
    Compute classification accuracy given raw model outputs (logits).

    logits : (N, num_classes) float tensor
    labels : (N,) int64 tensor of true class ids
    """
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.shape[0]


def plot_training_curves(history, save_path):
    """
    Plot training/validation loss and accuracy curves side by side
    and save the figure to `save_path`.

    history : dict with keys "train_loss", "val_loss", "train_acc", "val_acc",
              each a list of per-epoch values.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Loss vs. Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs. Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_model(model, save_path):
    """Save only the model's learnable parameters (state_dict)."""
    torch.save(model.state_dict(), save_path)


def load_model(model, load_path, map_location="cpu"):
    """Load previously saved parameters into an existing model instance."""
    state_dict = torch.load(load_path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model
