"""
Feature-only baseline: a plain 2-layer MLP with the same capacity as the
Vanilla GCN (same hidden size, same dropout), but with NO access to the
graph at all -- it classifies each paper using only its own word-features,
completely ignoring who cites whom.

This model is not part of the GCN architecture. It exists purely as a
baseline for compare_gcn_mlp.py, to measure how much the citation graph
itself contributes to classification accuracy, by holding every other
factor (architecture depth, hidden size, optimizer, regularization,
train/val/test split) fixed and changing only "graph-aware vs. not."
"""

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes)
        self.dropout = dropout

    def forward(self, node_features, adj=None):
        # `adj` is accepted and ignored only so this model has the same
        # call signature as GCN.forward() and can be dropped into the
        # existing train_model()/evaluate_model() functions unchanged.
        x = F.dropout(node_features, self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.fc2(x)
