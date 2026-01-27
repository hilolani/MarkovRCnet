from markovrcnet.io.load_matrix import load_adjacency
from markovrcnet.utils.sparse import SafeCSR


def _prepare_adj_matrix(adj_input, *, copy: bool = True):
    """
    Normalize adjacency matrix input.

    Parameters
    ----------
    adj_input : SafeCSR or path-like
        Input adjacency representation.
    copy : bool, default=True
        Whether to copy the input matrix.

    Returns
    -------
    SafeCSR
        Prepared adjacency matrix.
    """
    if isinstance(adj_input, SafeCSR):
        adj = adj_input
    else:
        adj = load_adjacency(adj_input)

    if copy:
        adj = adj.copy()

    return adj

