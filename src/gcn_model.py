"""
Vanilla Graph Convolutional Network (GCN), as introduced by
Kipf & Welling, "Semi-Supervised Classification with Graph
Convolutional Networks" (ICLR 2017).

The full derivation of every equation below is in
docs/mathematical_modelling.md. This file only contains the
standard 2-layer GCN -- no attention, no sampling, no residual
connections, no advanced tricks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolutionLayer(nn.Module):
    """
    A single graph convolution layer implementing:

        H_out = A_hat @ H_in @ W

    where:
        H_in  : (N, in_features)  node representations coming in
        A_hat : (N, N)            normalized adjacency matrix (precomputed, fixed)
        W     : (in_features, out_features) learnable weight matrix
        H_out : (N, out_features) node representations going out

    This is a direct implementation of Eq. (2) in the mathematical
    modelling document, without the nonlinearity (that is applied
    separately in the GCN class below so the layer stays reusable).
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        # A plain learnable weight matrix, same role as a Linear layer's weight.
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot/Xavier initialization keeps activations at a stable scale
        # across layers, which helps GCN training converge.
        nn.init.xavier_uniform_(self.weight)

    def forward(self, node_features, adj):
        # Step 1: mix each node's own features linearly -> (N, out_features)
        support = node_features @ self.weight
        # Step 2: aggregate over neighbours using the normalized adjacency
        output = adj @ support
        return output + self.bias


class GCN(nn.Module):
    """
    Standard 2-layer Vanilla GCN for node classification:

        Z = softmax( A_hat . ReLU( A_hat . X . W0 ) . W1 )

    Layer 1 (input -> hidden)  : GraphConvolutionLayer + ReLU + Dropout
    Layer 2 (hidden -> output) : GraphConvolutionLayer (raw class scores / logits)

    Dropout is applied to the input features of each layer during
    training only, which is the standard regularization used in the
    original GCN paper to prevent overfitting on small labeled sets.
    """

    def __init__(self, in_features, hidden_features, num_classes, dropout=0.5):
        super().__init__()
        self.gc1 = GraphConvolutionLayer(in_features, hidden_features)
        self.gc2 = GraphConvolutionLayer(hidden_features, num_classes)
        self.dropout = dropout

    def forward(self, node_features, adj):
        x = F.dropout(node_features, self.dropout, training=self.training)
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # Return raw logits; F.cross_entropy (used in train.py) applies
        # log-softmax internally, so we don't do it again here.
        return x
