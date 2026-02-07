"""
Utilities for PyTorch Geometric (PyG) interoperability.

This module provides lightweight conversion utilities to bridge
markovrcnet outputs with PyG-compatible data structures.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix


# --------------------------------------------------
# (A) CSR adjacency -> PyG edge_index
# --------------------------------------------------

def csr_to_edge_index(adj: csr_matrix) -> torch.Tensor:
    """
    Convert a CSR adjacency matrix to PyG edge_index format.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        Adjacency matrix.

    Returns
    -------
    edge_index : torch.LongTensor, shape [2, num_edges]
    """
    coo = adj.tocoo()
    edge_index = np.vstack((coo.row, coo.col))
    return torch.from_numpy(edge_index).long()


# --------------------------------------------------
# (B) MCL clusters -> node labels
# --------------------------------------------------

def clusters_to_node_labels(clusters: dict, num_nodes: int) -> torch.Tensor:
    """
    Convert MCL clustering result to node-wise labels.

    Parameters
    ----------
    clusters : dict
        {cluster_id: [node indices]}
    num_nodes : int

    Returns
    -------
    labels : torch.LongTensor, shape [num_nodes]
    """
    labels = np.full(num_nodes, -1, dtype=np.int64)
    for cid, nodes in clusters.items():
        labels[nodes] = cid
    return torch.from_numpy(labels)


# --------------------------------------------------
# (C) MiF / MiFDI -> node feature tensor
# --------------------------------------------------

def mifdi_to_node_features(
    mifdi_result: list,
    num_nodes: int,
    default: float = 0.0
) -> torch.Tensor:
    """
    Convert MiFDI output to PyG node feature tensor.

    Parameters
    ----------
    mifdi_result : list
        Output of mif.MiFDI()[0]
        Expected format: [(?, node_id, value), ...]
    num_nodes : int
    default : float
        Fill value for missing nodes.

    Returns
    -------
    x : torch.FloatTensor, shape [num_nodes, 1]
    """
    x = np.full((num_nodes, 1), default, dtype=np.float32)

    for _, node, value in mifdi_result:
        x[node, 0] = value

    return torch.from_numpy(x)


def adjmats_to_pyg_data(
    adj_csr,
    clusters=None,
    mifdi_raw=None,
):
    """
    Convert markovrcnet adjacency and results into PyG Data.

    Parameters
    ----------
    adj_csr : scipy.sparse.csr_matrix
    clusters : dict[int, list[int]], optional
        Output of mclprocess
    mifdi_raw : optional
        Output of MiFDI(...)[0]

    Returns
    -------
    torch_geometric.data.Data
    """
    edge_index = csr_to_edge_index(adj_csr)
    data = Data(edge_index=edge_index)

    if clusters is not None:
        data.y = clusters_to_node_labels(clusters, adj_csr.shape[0])

    if mifdi_raw is not None:
        data.x = mifdi_to_node_features(mifdi_raw, adj_csr.shape[0])

    return data

def mifdi_to_node_features(mifdi_raw, num_nodes):
    """
    mifdi_raw: output of MiFDI(...)[0]
               iterable of (cluster, node, value)
    """
    x = np.zeros(num_nodes, dtype=np.float32)
    for _, node, value in mifdi_raw:
        x[node] = value
    return torch.from_numpy(x).unsqueeze(1)
