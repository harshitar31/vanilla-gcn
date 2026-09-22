"""
Entry point for the Vanilla GCN pipeline on the Cora citation dataset.

Usage:
    python main.py
    python main.py --epochs 300 --hidden_dim 32 --lr 0.005

Running this script will:
    1. Load and preprocess the Cora graph.
    2. Build a 2-layer Vanilla GCN.
    3. Train it and record loss/accuracy history.
    4. Evaluate it on the held-out test split.
    5. Save a training-curve plot and the trained model weights.
"""

import argparse
import os

from src.data_loader import load_cora
from src.preprocessing import preprocess
from src.gcn_model import GCN
from src.train import train_model
from src.evaluate import evaluate_model, print_evaluation
from src.utils import set_seed, plot_training_curves, save_model


def parse_args():
    parser = argparse.ArgumentParser(description="Vanilla GCN on Cora")
    parser.add_argument("--data_dir", type=str, default="data/cora", help="Path to raw Cora files")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden layer size")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="L2 regularization strength")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Where to save plots/models")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    plots_dir = os.path.join(args.output_dir, "plots")
    models_dir = os.path.join(args.output_dir, "saved_models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ---- 1. Load raw data ----
    print("Loading Cora dataset...")
    features, labels, class_names, edges = load_cora(args.data_dir)
    print(f"  Nodes: {features.shape[0]}, Features: {features.shape[1]}, "
          f"Classes: {len(class_names)}, Edges: {edges.shape[0]}")

    # ---- 2. Preprocess into model-ready tensors ----
    data = preprocess(features, labels, edges, seed=args.seed)
    print(f"  Train nodes: {int(data['train_mask'].sum())}, "
          f"Val nodes: {int(data['val_mask'].sum())}, "
          f"Test nodes: {int(data['test_mask'].sum())}")

    # ---- 3. Build the Vanilla GCN ----
    model = GCN(
        in_features=data["features"].shape[1],
        hidden_features=args.hidden_dim,
        num_classes=len(class_names),
        dropout=args.dropout,
    )

    # ---- 4. Train ----
    print("\nTraining...")
    history = train_model(
        model, data,
        num_epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
    )

    # ---- 5. Evaluate on the test split ----
    results = evaluate_model(model, data, mask_name="test_mask", class_names=class_names)
    print_evaluation(results, split_name="Test")

    # ---- 6. Save training curves and model weights ----
    plot_path = os.path.join(plots_dir, "training_curves.png")
    plot_training_curves(history, plot_path)
    print(f"\nSaved training curves to {plot_path}")

    model_path = os.path.join(models_dir, "gcn_cora.pt")
    save_model(model, model_path)
    print(f"Saved trained model weights to {model_path}")


if __name__ == "__main__":
    main()
