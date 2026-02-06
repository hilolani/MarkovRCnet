# MarkovRCnet

MarkovRCnet, which stands for Markovian Refined Complex networks, is a Python package for analyzing network structure using Markov-based clustering and reachability measures, integrating MCL, refined MCL variants, and the MiF family of metrics.

# About

MarkovRCnet is a Python library (not a CLI tool), designed for research and experimental use.
It is not a deep learning framework.
Instead, it provides classical Markov-based graph clustering and influence metrics,
intended to be composed with other graph learning or GNN pipelines.

**Input**
- `scipy.sparse.csr_matrix` (adjacency matrix)

**Output**
- `dict[int, list[int]]` (cluster assignments)
- `float` (influence / score metrics)

# Quick Start

MarkovRCnet is a Python *library*.
Please install it in a virtual environment using `pip`.
`pipx` is **not recommended**, as this package is not a CLI tool.

## Setup

```bash
sudo apt install python3-pip python3-venv
python3 -m venv .venv
source .venv/bin/activate
```

Install

    pip install markovrcnet

Example

    from markovrcnet.datasets import load_all_adjmats
    from markovrcnet.mcl import mclprocess

    mats = load_all_adjmats()
    clusters = mclprocess(mats["karateclub"])
    print(clusters)


# Overview

MarkovRCNet is a Python library for analyzing complex networks using Markov random walk–based methods.

Many real-world systems—such as social, biological, and information networks—can be naturally modeled as graphs. Yet, understanding their global and local structure remains challenging due to sparsity, scale, and complex connectivity.

A natural way to explore a graph is to imagine a random walker moving from node to node. At each step, the walker selects its next position based solely on the current node, following a Markov process. Despite this simple rule, the resulting dynamics reveal rich structural information about the network.

MarkovRCNet leverages this principle to provide tools for:

measuring similarity and distance between nodes,

identifying communities and flow structures,

extracting interpretable representations of graphs.

By grounding network analysis in Markov dynamics, MarkovRCNet offers an intuitive yet mathematically principled framework—accessible to beginners and useful for advanced research, including applications in machine learning and graph-based AI.

# Two Components of This Library

This library consists of two components that are formally independent yet closely related: Re_MCL and MiF.

Re_MCL revisits and reimplements the Markov Cluster Algorithm (MCL), a graph clustering method proposed over two decades ago, while incorporating unique extensions and refinements.
MiF (Markov inverse F-measure), in contrast, provides a flexible framework for measuring distances and similarities between nodes based on the intrinsic characteristics of complex networks.

## Re_MCL: MCL and Recurrent MCL (RMCL)

The Markov Cluster Algorithm (MCL), proposed by Van Dongen, is a fast, scalable, and high-quality method for graph clustering (community detection) based on the simulation of random walks on graphs.

MCL models random walks using two simple algebraic matrix operations known as expansion and contraction. Expansion corresponds to standard matrix multiplication, while contraction combines Hadamard (element-wise) multiplication with rescaling. By alternately applying these two operations, MCL partitions a graph into a fixed number of disjoint subgraphs.

When applied to semantic networks—such as those constructed from word co-occurrence relationships—MCL often produces clusters that correspond to meaningful conceptual categories.

This library implements the conventional MCL algorithm and additionally provides an extended algorithm called Recurrent MCL (RMCL), originally developed at the former Akama Laboratory at Tokyo Institute of Technology. RMCL can be executed via the function rmcl_basic().

A key component of RMCL is Branching Markov Clustering (BMCL), which addresses a well-known limitation of standard MCL: severe cluster size imbalance. This issue becomes particularly pronounced in graphs exhibiting scale-free and heterophilic characteristics.

### Core Clusters and Core Hubs

In graphs whose degree distributions approximately follow a power law—such as semantic networks derived from large language corpora—standard MCL often produces one extremely large cluster alongside many small ones. We refer to this large cluster as the core cluster. The node with the highest degree within the core cluster is designated as the core hub.

BMCL enables the division of such oversized core clusters into appropriately sized subgraphs. It achieves this by introducing latent adjacency relationships between Markov clusters, thereby reconstructing a more informative network structure.

### Latent Adjacency Construction

To discover new relationships between Markov clusters, BMCL utilizes both the core cluster and smaller non-core clusters adjacent to it. From each Markov cluster, the node with the maximum degree is selected as a representative node (hub).

Using the adjacency matrix of the original graph, latent adjacency information is computed, and MCL is reapplied to this reconstructed adjacency matrix.

### Splitting Large Core Clusters

First, BMCL identifies existing connections between non-corehub nodes within the core cluster and representative hubs of other Markov clusters. It then determines whether two non-corehub nodes in the core cluster can be virtually connected via a representative hub of an external Markov cluster within two hops, and counts the number of such distinct paths.

These inferred connections are interpreted as latent adjacencies between node pairs inside the core cluster. To divide the core cluster into balanced subclusters, MCL is reapplied to the resulting latent adjacency matrix.

### Merging Small Non-Core Clusters (Reverse Branching)

BMCL also supports the merging of small non-core clusters through a process known as reverse branching. In this step, connections between the core hub and representative hubs of non-core clusters are examined.

BMCL determines whether two representative hubs can be connected via the core hub within two hops, and if so, how many distinct paths exist. These inferred connections are likewise treated as latent adjacency information. MCL is then applied to this latent adjacency matrix, enabling the merging of small non-core clusters into larger, more meaningful structures.

## MiF (Markov inverse F-measure)

While Re_MCL focuses on restructuring graph topology through Markov dynamics, MiF addresses a complementary problem: measuring similarity and distance within complex networks.

MiF (Markov inverse F-measure) is a similarity (distance) measure between vertices in a graph, originally proposed by Akama et al. (2015).

MiF evaluates how closely two vertices are related by modeling how information flows between them through a Markov random walk. Unlike conventional graph similarity measures, MiF integrates both local and global structural information into a single framework.

From a local perspective, MiF considers co-occurrence-based similarity, reflecting how strongly two vertices overlap in their immediate neighborhoods. This idea is conceptually related to measures such as the Jaccard and Simpson coefficients.

From a global perspective, MiF incorporates geodesic-based similarity, taking into account shortest path lengths and the number of such paths. MiF naturally balances these two perspectives, enabling robust similarity estimation even in complex network structures.

The MiF value is normalized to lie within the interval [0, 1], where larger values indicate stronger similarity.

### Parameterization and Network Characteristics

MiF includes several free parameters that allow the metric to adapt to different network characteristics.

In classical set-based similarity measures, normalization often relies on the size of the union of two sets. In graph-based settings, however, such normalization can become problematic due to degree imbalance, degree correlation, or scale-free structures.

To address this issue, MiF introduces a parameter β (0 < β < 1), which controls how vertex degrees contribute to normalization. By using a harmonic-mean–based formulation, MiF can emphasize or suppress degree effects. In practice, choosing β values close to zero allows heterophilic or homophilic properties of the network to be highlighted more clearly.

MiF also accounts for the influence of longer paths in random walks. While shorter paths usually dominate similarity, longer paths and detours may still carry meaningful structural information. This effect is controlled by another parameter α, which gradually decreases the contribution of paths as their length increases.

### MiF Degradation Index (MiFDI)

This package also introduces a derived metric called the MiF Degradation Index (MiFDI).

MiFDI analyzes how similarity degrades as a random walk expands from a selected starting vertex. A random walk is initiated from a specific vertex (for example, a vertex with minimal degree), and MiF values between the starting vertex and each visited vertex are computed.

These values are recorded on a logarithmic scale and averaged at each step of the walk. Depending on the configuration, self-loops can be included or excluded. When self-loops are excluded, propagation stops at a vertex once it has been reached.

MiFDI provides a compact representation of how rapidly relational similarity decays across the network.

# Usage

## Install

After publishing on PyPI, you can install MarkovRCnet via:

    pip install markovrcnet

(If you are using Google Colab, prefix `pip` with `!`.)

## Input

In this package, all public APIs accept either a file path or a SafeCSR object defined in utils.sparse for specifying the adjacency matrix of a graph. Not only Matrix Market format sparse matrices (.mtx files), but also sparse matrices in other formats, and even dense matrices (though not recommended), can be automatically converted into SafeCSR object by io.load_matrix. Dense matrices are supported for convenience, but sparse representations are strongly recommended for performance and memory efficiency.

Input normalization is handled consistently across MCL and MiF, and internal functions operate on SafeCSR objects only. Input normalization (path / SafeCSR) and copy control are handled exclusively at API entry points.

## MCL and MiF

This section demonstrates a minimal yet representative workflow of MarkovRCnet: loading an adjacency matrix, performing Markov Clustering (MCL), and analyzing intra- and inter-cluster reachability using MiF. The same workflow applies to RMCL and MiFDI, which extend MCL and MiF by incorporating refinement and directional information. See the following examples for their basic usage.

We use the classic Zachary’s Karate Club network as a small but non-trivial example. In this example, MiF values from a dangling node inside its own MCL cluster are significantly larger than those to nodes in the other cluster. This illustrates that MiF captures cluster-aware reachability consistent with the MCL partition.


    from markovrcnet.datasets import load_all_adjmats

    from markovrcnet.mcl import mclprocess

    from markovrcnet.mif import MiF

    from markovrcnet.io.load_matrix import load_adjacency

    import numpy as np

    import networkx as nx

    mats = load_all_adjmats()


    karatec = load_adjacency(mats["karateclub"])


    # Analysis of Karate Club

    Gobj = nx.from_scipy_sparse_array(karatec)

    diameter_karatec = nx.diameter(Gobj)

    print(f"The diameter of the Karate Club is: {diameter_karatec}")

    # The diameter of the Karate Club is: 5

    deglist = dict(nx.degree(Gobj))

    min_deg_node = min(deglist, key=deglist.get)

    print(f"The vertex number of the smallest degree is: {min_deg_node}")

    # The vertex number of the smallest degree is: 11


    # MCL

    result_karate_mcl = mclprocess(karatec)

    print(result_karate_mcl)

    # {0: [0, 1, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21], 1: [2, 8, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]}


    # MiF

    # Select as the starting node the vertex 11 belonging to the cluster number 0.

    # Remove the starting node itself from the target list

    result_karate_mcl[0].remove(min_deg_node)

    mif_clus0 = [MiF(karatec, min_deg_node, i, 0.5, diameter_karatec) for i in result_karate_mcl[0]]

    mif_clus1 = [MiF(karatec, min_deg_node, i, 0.5, diameter_karatec) for i in result_karate_mcl[1]]

    print(

        f"The mean MiF value of the cluster 0 from a dangling node inside is large: "

        f"{np.mean(mif_clus0)}"

    )

    #  0.03705702889419532

    print(

        f"The mean MiF value of cluster 1 from that exterior node is small: "

        f"{np.mean(mif_clus1)}"

    )

    #  0.0044844434956278776

    #  This illustrates that MiF captures cluster-aware reachability consistent with the MCL partition.


## The extension of workflow

The same workflow applies to RMCL and MiFDI, which extend MCL and MiF by incorporating refinement and directional information.

See the following examples for their basic usage.

## RMCL functions

The RMCL addresses a well-known limitation of standard MCL, i.e. issue of severe cluster size imbalance particularly pronounced in graphs exhibiting scale-free and heterophilic characteristics. Therefore, it is mainly applicable when the size (number of elements) of the largest cluster output as a result of MCL exceeds a certain threshold (by default, when the number of elements per cluster exceeds twice the standard deviation).

### Branching_Rmcl

The branching_rmcl function enables the division of such oversized core clusters into appropriately sized subgraphs. It achieves this by introducing latent adjacency relationships between Markov clusters, thereby reconstructing a more informative network structure.

As suitable for the demonstration purpose, the following example doesn't use Karate club network, but "scalefree.mtx" (stored in data repository), which was created based on the Barabási–Albert (BA) model, starting with a complete graph of 5 nodes for which additional nodes with degree 5 repeatedly underwent preferential attachment until the total number of vertices reaches 100. The branching_rmcl() also supports the merging of small, non-core clusters through a process called reverse branching.

### SRmcl and Mixed Rmcl

Simply-repeated MCL, abbreviated as srmcl, operates as follows: if a core cluster exists, it extracts the hub with the highest degree from within it and then re-runs MCL iteratively only on the core cluster.

The mixed_rmcl applies srmcl to the core cluster and, for non-core ones, applies reverse branching rmcl.


    # Scale free graph.

    from markovrcnet.datasets import load_all_adjmats

    import markovrcnet.mcl as mcl


    mats = load_all_adjmats()

    sf = mats["scalefree"]

    cluslist = mcl.mclprocess(sf)

    print(f"The MCL result of 'scalefree' graph is: {cluslist}")

    # The MCL result of 'scalefree' graph is: {0: [0, 11, 18, 21, 49, 80], 1: [3, 48, 57, 72], 2: [1, 2, 4, 5, 9, 10, 12, 13, 14, 15, 16, 17, 23, 24, 25, 26, 28, 29, 33, 37, 39, 41, 43, 45, 50, 51, 52, 54, 58, 59, 60, 61, 62, 63, 68, 70, 75, 79, 85, 86, 88, 89, 90, 92, 93, 96, 98], 3: [6, 46], 4: [7, 66, 77, 97], 5: [8, 27, 32, 42, 53, 55, 56, 64, 67, 76, 78, 83, 84, 87, 94, 95], 6: [19, 30, 69], 7: [20, 38], 8: [22, 47, 81], 9: [31], 10: [35], 11: [36, 40, 65], 12: [44], 13: [73], 14: [82], 15: [34, 71, 74, 91, 99]}


    result_branching = mcl.branching_rmcl(cluslist, sf)

    print(f"The Branching RMCL result of 'scalefree' graph under the default settings is: {result_branching}")

    # {0: [1, 2, 5, 9, 10, 12, 13, 14, 15, 16, 17, 23, 24, 25, 28, 29, 33, 37, 39, 41, 45, 50, 51, 52, 54, 58, 59, 60, 61, 62, 63, 68, 70, 85, 88, 90, 92, 93, 98], 1: [26], 2: [43], 3: [75], 4: [79], 5: [86], 6: [89], 7: [96], 8: [4]}


    result_sr = mcl.sr_mcl(cluslist, sf)

    print(f"The SR MCL result under the default settings is : {result_sr}")

    # {0: [1, 2, 4, 5, 9, 10, 12, 13, 14, 15, 17, 24, 25, 26, 28, 29, 33, 37, 39, 41, 50, 52, 54, 60, 61, 62, 63, 68, 75, 79, 85, 86, 88, 92, 98], 1: [16, 89], 2: [23, 93], 3: [43, 58], 4: [45, 70, 90, 96], 5: [51, 59]}


    result_mixed =  mcl.mixed_rmcl(cluslist, sf, threspruning = 3.0, branching = False)

    print(f"The Mixed MCL result under the default settings but changing the threshold of latent adjacency weight into 3.0 and without applying reverse branching to non-core clusters is : {mcl.mcllist_to_mcldict(result_mixed)}")

    # {0: [0, 11, 18, 21, 49, 80], 1: [1, 2, 4, 5, 9, 10, 12, 13, 14, 15, 17, 24, 25, 26, 28, 29, 33, 37, 39, 41, 50, 52, 54, 60, 61, 62, 63, 68, 75, 79, 85, 86, 88, 92, 98], 2: [3, 48, 57, 72], 3: [6, 46], 4: [7, 66, 77, 97], 5: [8, 27, 32, 42, 53, 55, 56, 64, 67, 76, 78, 83, 84, 87, 94, 95], 6: [16, 89], 7: [19, 30, 69], 8: [20, 38], 9: [22, 47, 81], 10: [23, 93], 11: [31], 12: [34, 71, 74, 91, 99], 13: [35], 14: [36, 40, 65], 15: [43, 58], 16: [44], 17: [45, 70, 90, 96], 18: [51, 59], 19: [73], 20: [82]}

## Docker

### Docker as Reproducible environment

You can also run MarkovRCnet using Docker without installing Python dependencies.

```bash
docker pull akamahilolani/markovrcnet:latest
docker run --rm akamahilolani/markovrcnet:latest \
  python -c "from markovrcnet.mif import MiF; print(MiF)"
```

### Jupyter Docker

```bash
docker pull akamahilolani/markovrcnet-jupyter:latest
docker run --rm -it -p 10001:10001 akamahilolani/markovrcnet-jupyter:latest
```
Open the printed URL in your browser.

## Notes

For more detailed information about the options for the functions introduced here, or for usage of other functions, please refer to the following web page.

    https://sites.google.com/site/akamatitechlab/markovrcnet


# References

Stijn van Dongen, Graph Clustering by Flow Simulation, 2000 https://dspace.library.uu.nl/bitstream/handle/1874/848/full.pdf?sequence=1&isAllowed=y


Jaeyoung Jung and Hiroyuki Akama. 2008. Employing Latent Adjacency for Appropriate Clusteringof Semantic Networks. New Trends in Psychometrics p.131-140


Hiroyuki Akama et al, 2008. Random graph model simulations of semantic networks for associative Concept dictionaries, TextGraphs-3 doi: https://dl.acm.org/doi/10.5555/1627328.1627337


Hiroyuki Akama et al., 2008. How to Take Advantage of the Limitations with Markov Clustering?--The Foundations of Branching Markov Clustering (BMCL), IJCNLP-2008, p.901~906 https://aclanthology.org/I08-2129.pdf


Hiroyuki Akama et al., 2007. Building a clustered semantic network for an Entire Large Dictionary of Japanese, PACLING-2007, p.308~316 https://www.researchgate.net/publication/228950233_Building_a_clustered_semantic_network_for_an_Entire_Large_Dictionary_of_Japanese


Jaeyoung Jung, Maki Miyake, Hiroyuki Akama. 2006. Recurrent Markov Cluster (RMCL) Algorithm for the Refinement of the Semantic Network. In: LREC. p. 1428–1431 http://www.lrec-conf.org/proceedings/lrec2006/


Hiroyuki Akama, Maki Miyake, Jaeyoung Jung, Brian Murphy, 2015. Using Graph Components Derived from an Associative Concept Dictionary to Predict fMRI Neural Activation Patterns that Represent the Meaning of Nouns, PLoS ONE, doi: https://doi.org/10.1371/journal.pone.0125725
