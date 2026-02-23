"""
generate_figures.py

Reproduces all figures used in:
"Revisiting Markov Chain-Based Complex Network Analysis:
 Introducing MarkovRCnet as a Practical Tool"

Usage
-----
From the project root:

    python data/generate_figures.py

Outputs
-------
All figures are saved in:
    data/figures/

Interactive visualization:
    karate_colormap_mifdi.html
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyvis")

# Optional IPython display (works without Jupyter)
try:
    from IPython.display import HTML, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.sparse import coo_matrix
from scipy.spatial.distance import squareform

import networkx as nx
from pyvis.network import Network

import markovrcnet.mcl as mcl
from markovrcnet.mcl import mcllist_to_mcldict
import markovrcnet.mif as mif
from markovrcnet.datasets import load_all_adjmats
from markovrcnet.io import load_adjacency


# =====================================================
# Paths
# =====================================================
BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)


# =====================================================
# Data preparation
# =====================================================
def prep_karate():
    mats = load_all_adjmats()
    karatec = mats["karateclub"]

    comb = [(i, j) for i in range(34) for j in range(34) if i < j]
    resultlist = []

    for i, j in comb:
        val = mif.MiF(karatec, i, j, 0.5, 5)
        resultlist.append((i, j, float(val)))

    resultdi = mif.MiFDI(karatec, startingvertices="min")
    mifdivallist = [x[2] for x in resultdi[0]]

    keys = np.array(mifdivallist)
    G = nx.karate_club_graph()

    return resultlist, keys, G


def prep_scalefree():
    mats = load_all_adjmats()
    sf = mats["scalefree"]

    # MCL
    cluslist = mcl.mclprocess(sf)

    # RMCL
    result_mixed = mcl.mixed_rmcl(
        cluslist,
        sf,
        threspruning=2.0,
        branching=True
    )
    dict_result = mcllist_to_mcldict(result_mixed)

    mcl_clusters = list(cluslist.values())
    rmcl_clusters = list(dict_result.values())

    mcl_titles = [f"cluster{i}" for i in range(len(mcl_clusters))]
    rmcl_titles = [f"cluster{i}" for i in range(len(rmcl_clusters))]

    # Core cluster
    max_size = max(len(c) for c in mcl_clusters)
    core_nodes = [c for c in mcl_clusters if len(c) == max_size][0]
    noncore_nodes = [i for i in range(100) if i not in core_nodes]

    # MiFDI
    resultdi = mif.MiFDI(sf, startingvertices="max")
    vals = [x[2] for x in resultdi[0]]
    shift = abs(min(vals)) + 0.1
    mifvals = [v + shift for v in vals]

    # Graph object
    sf_obj = load_adjacency(sf)
    G = nx.from_scipy_sparse_array(sf_obj)

    return (
        mcl_clusters, mcl_titles,
        rmcl_clusters, rmcl_titles,
        core_nodes, noncore_nodes,
        mifvals, G
    )


# =====================================================
# Figure functions
# =====================================================
def fig1(resultlist):
    rows = [t[0] for t in resultlist]
    cols = [t[1] for t in resultlist]
    vals = [t[2] for t in resultlist]

    rows_sym = rows + cols
    cols_sym = cols + rows
    vals_sym = vals + vals

    n = max(rows_sym) + 1
    sparse_mat = coo_matrix((vals_sym, (rows_sym, cols_sym)), shape=(n, n))
    dense = sparse_mat.toarray()

    dist = 1 - dense
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist)

    Z = linkage(condensed, method="average")
    order = leaves_list(Z)

    reordered = dense[order][:, order]

    plt.figure(figsize=(6, 6))
    plt.imshow(reordered, cmap="plasma")
    plt.colorbar()
    plt.xticks(np.arange(len(order)), order, rotation=90)
    plt.yticks(np.arange(len(order)), order)
    plt.tight_layout()

    out = OUT_DIR / "fig1_mif_heatmap.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def fig2(keys, G):
    cmap = cm.get_cmap("coolwarm")
    hexcolors = []

    for i in range(len(keys)):
        norm = 1 - (i / (len(keys) - 1))
        rgba = cmap(norm)
        hexcolors.append(mcolors.rgb2hex(rgba[:3]))

    idx = np.argsort(np.argsort(-keys))
    sorted_colors = np.array(hexcolors)[idx]

    for node in G.nodes():
        G.nodes[node]["label"] = str(node)
        G.nodes[node]["size"] = 10 + G.degree(node) * 3
        G.nodes[node]["color"] = sorted_colors[node]

    net = Network(
        notebook=False,
        cdn_resources="in_line",
        height="700px",
        width="100%"
    )

    net.from_nx(G)

    out_html = OUT_DIR / "karate_colormap_mifdi.html"
    net.write_html(str(out_html))
    print(f"Saved: {out_html}")

    if IPYTHON_AVAILABLE:
        display(HTML(filename=str(out_html)))


def _draw_cluster_panel(panel, clusters, titles, core_nodes, mifvals, G, filename):
    subgraphs = [G.subgraph(c).copy() for c in clusters]
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    pos = nx.spring_layout(G, seed=42)

    for ax, sg, title in zip(axes, subgraphs, titles):
        if panel == "upper":
            sizes = [G.degree(n) * 20 for n in sg.nodes()]
        else:
            sizes = [mifvals[n] * 200 for n in sg.nodes()]

        colors = [
            "lightgreen" if n in core_nodes else "lightblue"
            for n in sg.nodes()
        ]

        nx.draw_networkx(
            sg,
            pos=pos,
            ax=ax,
            node_size=sizes,
            node_color=colors,
            with_labels=False
        )
        ax.set_title(title)

    for ax in axes[len(subgraphs):]:
        ax.axis("off")

    plt.tight_layout()
    out = OUT_DIR / filename
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def fig3_and_4(data):
    (
        mcl_clusters, mcl_titles,
        rmcl_clusters, rmcl_titles,
        core_nodes, noncore_nodes,
        mifvals, G
    ) = data

    _draw_cluster_panel(
        "upper", mcl_clusters, mcl_titles,
        core_nodes, mifvals, G,
        "fig3_mcl_degree.png"
    )

    _draw_cluster_panel(
        "lower", mcl_clusters, mcl_titles,
        core_nodes, mifvals, G,
        "fig3_mcl_mifdi.png"
    )

    _draw_cluster_panel(
        "upper", rmcl_clusters, rmcl_titles,
        core_nodes, mifvals, G,
        "fig4_rmcl_degree.png"
    )

    _draw_cluster_panel(
        "lower", rmcl_clusters, rmcl_titles,
        core_nodes, mifvals, G,
        "fig4_rmcl_mifdi.png"
    )


# =====================================================
# Main
# =====================================================
def main():
    print("Generating all figures...")

    # Karate
    resultlist, keys, Gk = prep_karate()
    fig1(resultlist)
    fig2(keys, Gk)

    # Scale-free
    data = prep_scalefree()
    fig3_and_4(data)

    print("Done.")
    print(f"All outputs saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()