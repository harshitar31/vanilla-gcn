"""
Evaluation pipeline for the Vanilla GCN.

Computes standard classification metrics on the held-out test set:
accuracy, macro-averaged precision/recall/F1, and a confusion matrix.
Macro-averaging treats every class equally regardless of how many
papers it has, which is informative here since Cora's 7 classes are
not perfectly balanced.
"""

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, data, mask_name="test_mask", class_names=None):
    """
    Run the model in inference mode on the given split and return a dict
    of metrics plus the raw predictions (useful for further inspection).
    """
    model.eval()
    with torch.no_grad():
        logits = model(data["features"], data["adj"])
        predictions = logits.argmax(dim=1)

    mask = data[mask_name]
    y_true = data["labels"][mask].numpy()
    y_pred = predictions[mask].numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def print_evaluation(results, split_name="Test"):
    """Pretty-print the metrics returned by evaluate_model()."""
    print(f"\n{split_name} Set Evaluation")
    print("-" * 40)
    print(f"Accuracy           : {results['accuracy']:.4f}")
    print(f"Precision (macro)  : {results['precision_macro']:.4f}")
    print(f"Recall (macro)     : {results['recall_macro']:.4f}")
    print(f"F1-score (macro)   : {results['f1_macro']:.4f}")
    print("\nPer-class report:")
    print(results["classification_report"])
    print("Confusion matrix (rows = true class, columns = predicted class):")
    print(results["confusion_matrix"])
