"""
Data loading for the Cora citation dataset.

Cora contains 2708 scientific papers (nodes) split into 7 classes.
Each paper is described by a 1433-dimensional binary word vector
(1 = the corresponding word from the dictionary is present in the
paper, 0 = absent). Papers are connected by citation links (edges).

Raw files (downloaded from https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz):
    cora.content : one line per paper -> <paper_id> <1433 binary features> <class_label>
    cora.cites   : one line per citation -> <cited_paper_id> <citing_paper_id>
"""

import os
import numpy as np


def load_cora(data_dir):
    """
    Read cora.content and cora.cites and return raw numpy arrays.

    Returns:
        features    : (N, F) float32 array of binary word features
        labels      : (N,) int64 array of class ids in [0, num_classes)
        class_names : list of class name strings, indexed by label id
        edges       : (E, 2) int64 array of (source, target) node-index pairs
    """
    content_path = os.path.join(data_dir, "cora.content")
    cites_path = os.path.join(data_dir, "cora.cites")

    # --- Read cora.content: paper_id, 1433 word-features, class label ---
    paper_ids = []
    features = []
    labels_raw = []

    with open(content_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            paper_ids.append(parts[0])                       # first column: paper id
            features.append([int(x) for x in parts[1:-1]])   # middle columns: word features
            labels_raw.append(parts[-1])                     # last column: class name (string)

    features = np.array(features, dtype=np.float32)

    # Map paper_id (string) -> row index, since cora.cites refers to papers by id
    id_to_index = {pid: i for i, pid in enumerate(paper_ids)}

    # Map class name (string) -> integer label, in sorted order for reproducibility
    class_names = sorted(set(labels_raw))
    class_to_index = {name: i for i, name in enumerate(class_names)}
    labels = np.array([class_to_index[name] for name in labels_raw], dtype=np.int64)

    # --- Read cora.cites: citation edges between paper ids ---
    edges = []
    with open(cites_path, "r") as f:
        for line in f:
            cited, citing = line.strip().split("\t")
            # Some ids in cora.cites do not appear in cora.content; skip those.
            if cited in id_to_index and citing in id_to_index:
                edges.append((id_to_index[cited], id_to_index[citing]))

    edges = np.array(edges, dtype=np.int64)

    return features, labels, class_names, edges
