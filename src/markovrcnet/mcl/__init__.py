"""
Markov Cluster-based algorithms (MCL / RMCL).

This module provides:
- Standard MCL
- Core-cluster analysis
- Recursive / branching MCL variants
"""

# --- Core algorithms (main entry points) ---
from .core import (
    mclprocess,
    get_soft_clusters_proto,
)

# --- Core-cluster / RMCL family ---
from .core import (
    coreclusQ,
    mclus_analysis,
    branching_rmcl,
    sr_mcl,
    mixed_rmcl,
    rmcl_basic,
)

# --- Conversion & helper utilities (mcl-specific) ---
from .core import (
    mcldict_to_mclset,
    mclset_to_mcldict,
    mcldict_to_mcllist,
    mcllist_to_mclset,
    mcllist_to_mcldict,
    mclset_to_mcllist,
    clusinfo_from_nodes,
)

__all__ = [
    # main
    "mclprocess",
    "get_soft_clusters_proto",

    # core / rmcl
    "coreclusQ",
    "mclus_analysis",
    "branching_rmcl",
    "sr_mcl",
    "mixed_rmcl",
    "rmcl_basic",

    # helpers
    "mcldict_to_mclset",
    "mclset_to_mcldict",
    "mcldict_to_mcllist",
    "mcllist_to_mclset",
    "mcllist_to_mcldict",
    "mclset_to_mcllist",
    "clusinfo_from_nodes",
]
