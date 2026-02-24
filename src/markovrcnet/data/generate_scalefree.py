"""
generate_scalefree.py

Generates the scale-free network used in this study.

Usage
-----
From the project root:

    python data/generate_scalefree.py

Output
------
data/scalefree.npz


Reproducibility Note
--------------------
The network is generated using a preferential attachment process
(degree-proportional selection), implemented in Python.

The original network used in the paper was generated using the following
Wolfram Mathematica code (archival source):

    scaleFreeNetwork[loop_Integer, mO_Integer: 3, m_Integer: 3] :=
        Module[{g, addlst, addNeVertex},
            (*Definition of a local function*)
            addNewVertex[grph_Graph] := Module[{selectPosition, n, problst, gr},
            n= V[grph];
            problst = Map[Total, ToAdjacencyMatrix[grph]] // #/Total[#] & // FoldList[Plus, 0, #] &; 
            selectPosition[prlst_List] := Module[{rv}, rv = Random[];
                Position[prlst, First[Select[prlst, GreaterEqual[#, rv] &, 1]], 1][[1, 1]] -1
                ];
            addlst={};
        While[Length[addlst] < m,
　　　　    addlst = (selectPosition[problst] // If[MemberQ[addlst, #], Append[addlst, #], addlst] &)]; 
            gr =AddVertex[grph, (n - m0) * {Random[], Random []}]; 
            AddEdges[gr, (Map[List[#, V[gr]]&, addlst])]
       ]:

    (*Initialization of a graph by creating a complete graph with m0 vertices*)
    g = CompleteGraph[mO];

    (*Generate an empty list for the vertices as connection targets.*) 
    addlst = {};
    While[Length[addlst] < m,
        addlst = (Random [Integer, {1, m0}] // If[!MemberQ [addlst, #], Append[addlst, #], addlst] &)]; 
        g = AddVertex[g, {0, 0}];
        g = AddEdges[g, Map[List[#, mO + 1] &, addlst]];
        NestList[Evaluate[AddNewVertex[#]]&, g, loop-1]
    ];

Therefore, the graph produced here follows the same generative principle,
but node identities and exact topology may differ due to differences in
random number generation and implementation details.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp
import random


# =====================================================
# Parameters (match paper settings)
# =====================================================
N = 100        # total nodes
m0 = 5         # initial complete graph size
m = 5          # edges added per new node
SEED = 42


# =====================================================
# Paths
# =====================================================
BASE_DIR = Path(__file__).parent
OUT_FILE = BASE_DIR / "scalefree.npz"


# =====================================================
# Preferential attachment (Mathematica-style)
# =====================================================
def generate_scalefree(n_nodes, m0, m):
    """
    Preferential attachment graph generation.

    Probability of selecting a node is proportional to its degree.
    """

    # Initial complete graph
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)

    for i in range(m0):
        for j in range(i + 1, m0):
            adj[i, j] = 1
            adj[j, i] = 1

    degrees = adj.sum(axis=1).tolist()

    # Add new nodes
    for new_node in range(m0, n_nodes):

        # Degree-proportional probabilities
        total_degree = sum(degrees[:new_node])
        probs = [degrees[i] / total_degree for i in range(new_node)]

        targets = set()
        while len(targets) < m:
            chosen = random.choices(range(new_node), weights=probs, k=1)[0]
            targets.add(chosen)

        # Add edges
        for t in targets:
            adj[new_node, t] = 1
            adj[t, new_node] = 1
            degrees[t] += 1

        degrees[new_node] = m

    return adj


# =====================================================
# Main
# =====================================================
def main():
    print("Generating scale-free network...")
    random.seed(SEED)
    np.random.seed(SEED)

    adj = generate_scalefree(N, m0, m)

    sparse_adj = sp.csr_matrix(adj)
    sp.save_npz(OUT_FILE, sparse_adj)

    print(f"Saved: {OUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
