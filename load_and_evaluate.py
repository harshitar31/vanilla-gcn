"""
Demonstrates loading a previously saved GCN checkpoint and evaluating it,
without re-training. Run main.py first so outputs/saved_models/gcn_cora.pt
exists.

Usage:
    python load_and_evaluate.py
"""

import argparse

from src.data_loader import load_cora
from src.preprocessing import preprocess
from src.gcn_model import GCN
from src.evaluate import evaluate_model, print_evaluation
from src.utils import set_seed, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Load a saved Vanilla GCN and evaluate it")
    parser.add_argument("--data_dir", type=str, default="data/cora")
    parser.add_argument("--model_path", type=str, default="outputs/saved_models/gcn_cora.pt")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Must match the saved model")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    features, labels, class_names, edges = load_cora(args.data_dir)
    data = preprocess(features, labels, edges, seed=args.seed)

    # Rebuild the exact same architecture used during training,
    # then load the trained weights into it.
    model = GCN(
        in_features=data["features"].shape[1],
        hidden_features=args.hidden_dim,
        num_classes=len(class_names),
    )
    model = load_model(model, args.model_path)
    print(f"Loaded model weights from {args.model_path}")

    results = evaluate_model(model, data, mask_name="test_mask", class_names=class_names)
    print_evaluation(results, split_name="Test (loaded model)")


if __name__ == "__main__":
    main()
