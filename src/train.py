"""
Training pipeline for the Vanilla GCN.

Cora is used in the transductive, semi-supervised setting: the model
sees the ENTIRE graph (all 2708 nodes and all edges) on every forward
pass, but the loss is only computed on the labeled nodes belonging to
the current split (train_mask). This is standard for GCN node
classification -- the graph structure of validation/test nodes is
still useful "context" for making predictions, we just never let
their labels leak into the loss or the gradient.
"""

import torch
import torch.nn.functional as F

from src.utils import accuracy


def train_model(model, data, num_epochs=200, lr=0.01, weight_decay=5e-4, verbose=True):
    """
    Train `model` on the preprocessed graph in `data` (see preprocessing.py
    for its keys: features, adj, labels, train_mask, val_mask, test_mask).

    Loss: cross-entropy between predicted class logits and true labels,
    computed ONLY on nodes in train_mask.
    Optimizer: Adam with L2 weight decay (matches the original GCN paper).

    Returns:
        history : dict of per-epoch train/val loss and accuracy lists.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        # ---- one training step ----
        model.train()
        optimizer.zero_grad()

        logits = model(data["features"], data["adj"])  # forward pass on the FULL graph
        train_loss = F.cross_entropy(logits[data["train_mask"]], data["labels"][data["train_mask"]])

        train_loss.backward()   # backpropagation: compute d(loss)/d(weights)
        optimizer.step()        # gradient descent update (Adam rule)

        # ---- evaluate on train/val sets (no gradient needed) ----
        model.eval()
        with torch.no_grad():
            logits = model(data["features"], data["adj"])
            val_loss = F.cross_entropy(logits[data["val_mask"]], data["labels"][data["val_mask"]])
            train_acc = accuracy(logits[data["train_mask"]], data["labels"][data["train_mask"]])
            val_acc = accuracy(logits[data["val_mask"]], data["labels"][data["val_mask"]])

        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss.item())
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss.item():.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}"
            )

    return history
