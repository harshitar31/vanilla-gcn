"""
Graph preprocessing for the Vanilla GCN.

This module turns the raw (features, labels, edges) produced by
data_loader.py into the tensors the GCN actually consumes:

    1. Build a symmetric adjacency matrix A from the citation edges.
    2. Add self-loops and apply symmetric normalization to get
       the normalized adjacency matrix A_hat used in the GCN
       propagation rule (see docs/mathematical_modelling.md, Section 3).
    3. Row-normalize the node features so each paper's word-vector
       sums to 1 (standard practice for bag-of-words features).
    4. Create train / validation / test masks with a stratified split.

Everything is kept as dense numpy/torch arrays (not sparse tensors)
because Cora is small (2708 nodes) and dense matrices are far easier
to inspect and reason about during a viva than sparse formats.
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def build_adjacency_matrix(edges, num_nodes):
    """
    Build a dense, symmetric, binary adjacency matrix A from a list of edges.

    Citations in cora.cites are directed (cited -> citing), but for a
    Vanilla GCN we treat the citation graph as undirected: if paper i
    cites or is cited by paper j, we set A[i, j] = A[j, i] = 1.
    """
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src, dst in edges:
        A[src, dst] = 1.0
        A[dst, src] = 1.0  # symmetric: make the graph undirected
    return A


def normalize_adjacency(A):
    """
    Compute the symmetrically normalized adjacency matrix used by the GCN:

        A_hat = D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2)

    where A_tilde = A + I (self-loops added so a node's own features
    contribute to its next-layer representation) and D_tilde is the
    diagonal degree matrix of A_tilde, i.e. D_tilde[i, i] = sum_j A_tilde[i, j].

    This matches Kipf & Welling (2017), Eq. 2.
    """
    num_nodes = A.shape[0]
    A_tilde = A + np.eye(num_nodes, dtype=np.float32)          # add self-loops

    degrees = A_tilde.sum(axis=1)                              # D_tilde diagonal entries
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0                      # guard isolated nodes (degree 0)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return A_hat.astype(np.float32)


def normalize_features(X):
    """
    Row-normalize the feature matrix so every row (paper) sums to 1.
    This prevents papers with unusually long word-vectors from
    dominating the propagation purely due to scale.
    """
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero for all-zero rows
    return X / row_sums


def make_split_masks(labels, train_size=0.6, val_size=0.2, seed=42):
    """
    Create boolean train/val/test masks with a stratified random split,
    i.e. each split preserves the overall class proportions.

    train_size + val_size + test_size = 1.0 (test_size is the remainder).
    """
    num_nodes = labels.shape[0]
    indices = np.arange(num_nodes)

    # First split off the training set...
    train_idx, rest_idx = train_test_split(
        indices, train_size=train_size, stratify=labels, random_state=seed
    )
    # ...then split the remainder into validation and test sets.
    val_fraction_of_rest = val_size / (1.0 - train_size)
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=val_fraction_of_rest, stratify=labels[rest_idx], random_state=seed
    )

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def preprocess(features, labels, edges, seed=42):
    """
    Run the full preprocessing pipeline and return ready-to-use torch tensors.

    Returns a dict with keys:
        features   : (N, F) float32 tensor, row-normalized
        adj        : (N, N) float32 tensor, normalized adjacency A_hat
        labels     : (N,) int64 tensor
        train_mask, val_mask, test_mask : (N,) bool tensors
    """
    num_nodes = features.shape[0]

    A = build_adjacency_matrix(edges, num_nodes)
    A_hat = normalize_adjacency(A)
    X = normalize_features(features)
    train_mask, val_mask, test_mask = make_split_masks(labels, seed=seed)

    return {
        "features": torch.tensor(X, dtype=torch.float32),
        "adj": torch.tensor(A_hat, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "train_mask": torch.tensor(train_mask, dtype=torch.bool),
        "val_mask": torch.tensor(val_mask, dtype=torch.bool),
        "test_mask": torch.tensor(test_mask, dtype=torch.bool),
    }
