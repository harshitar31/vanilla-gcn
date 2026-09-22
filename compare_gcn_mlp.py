"""
Original analysis: does the citation graph actually help?

Trains two models with identical hyperparameters, the identical
train/val/test split, and the identical training loop -- the ONLY
difference between them is that the GCN aggregates information across
the citation graph (via A_hat) while the MLP baseline classifies each
paper from its own word-features alone, with no knowledge of the graph
at all. Everything else (hidden size, dropout, optimizer, learning rate,
weight decay, epochs, random seed, data split) is held fixed.

This isolates the specific contribution of graph structure from every
other design choice, which the original GCN paper does not report
directly (it only reports GCN's own accuracy).

Usage:
    python compare_gcn_mlp.py
"""

import argparse
import os

import matplotlib.pyplot as plt

from src.data_loader import load_cora
from src.preprocessing import preprocess
from src.gcn_model import GCN
from src.mlp_model import MLP
from src.train import train_model
from src.evaluate import evaluate_model, print_evaluation
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Vanilla GCN vs. a feature-only MLP baseline")
    parser.add_argument("--data_dir", type=str, default="data/cora")
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def run_model(model_name, model, data, args):
    print(f"\nTraining {model_name}...")
    history = train_model(
        model, data,
        num_epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        verbose=False,
    )
    results = evaluate_model(model, data, mask_name="test_mask")
    print_evaluation(results, split_name=f"Test ({model_name})")
    return history, results


def main():
    args = parse_args()

    # Both models see the EXACT same preprocessed graph and split --
    # preprocess() is seeded, so this call is identical for both runs.
    set_seed(args.seed)
    features, labels, class_names, edges = load_cora(args.data_dir)
    data = preprocess(features, labels, edges, seed=args.seed)
    num_classes = len(class_names)

    # Reset the seed before building/training each model so both start
    # from the same random weight initialization conditions too --
    # the only thing that differs between the two runs is the architecture.
    set_seed(args.seed)
    gcn = GCN(data["features"].shape[1], args.hidden_dim, num_classes, args.dropout)
    gcn_history, gcn_results = run_model("Vanilla GCN (graph-aware)", gcn, data, args)

    set_seed(args.seed)
    mlp = MLP(data["features"].shape[1], args.hidden_dim, num_classes, args.dropout)
    mlp_history, mlp_results = run_model("MLP baseline (features only)", mlp, data, args)

    # ---- Summary table ----
    print("\n" + "=" * 50)
    print(f"{'Metric':<20}{'GCN':>12}{'MLP':>12}{'Gain':>12}")
    print("-" * 50)
    for key, label in [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precision"),
        ("recall_macro", "Recall"),
        ("f1_macro", "F1-score"),
    ]:
        gain = gcn_results[key] - mlp_results[key]
        print(f"{label:<20}{gcn_results[key]:>12.4f}{mlp_results[key]:>12.4f}{gain:>+12.4f}")
    print("=" * 50)

    # ---- Comparison plot ----
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    epochs = range(1, args.epochs + 1)
    axes[0].plot(epochs, gcn_history["val_acc"], label="GCN (uses graph)")
    axes[0].plot(epochs, mlp_history["val_acc"], label="MLP (features only)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Accuracy")
    axes[0].set_title("Validation Accuracy vs. Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = range(len(metrics))
    width = 0.35
    axes[1].bar([i - width / 2 for i in x], [gcn_results[m] for m in metrics], width, label="GCN")
    axes[1].bar([i + width / 2 for i in x], [mlp_results[m] for m in metrics], width, label="MLP")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(metric_labels)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Test Set: GCN vs. MLP")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    plot_path = os.path.join(plots_dir, "gcn_vs_mlp.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
