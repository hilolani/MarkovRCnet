import os, json, pickle
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix, coo_matrix, issparse

from markovrcnet.utils.logging import resolve_logger
from markovrcnet.utils.sparse import SafeCSR


def load_adjacency(adjacencymatrix, logger=None):
    """
    Load an adjacency matrix from a file path or array-like object
    and return it as SafeCSR.
    """
    log = resolve_logger(logger, "matrix")

    path_or_matrix = adjacencymatrix

    if isinstance(path_or_matrix, str) and os.path.exists(path_or_matrix):
        path = path_or_matrix
        ext = os.path.splitext(path)[1].lower()

        if ext == ".mtx":
            matrix = mmread(path).tocsr()
            log.info("Loaded .mtx → CSR")

        elif ext == ".npz":
            loaded = np.load(path, allow_pickle=True)
            if {'data', 'indices', 'indptr'}.issubset(loaded.files):
                matrix = csr_matrix(
                    (loaded['data'], loaded['indices'], loaded['indptr'])
                )
            else:
                matrix = csr_matrix(loaded['arr_0'])
            log.info("Loaded .npz → CSR")

        elif ext == ".pkl":
            with open(path, "rb") as f:
                matrix = load_adjacency(pickle.load(f), logger=log)

        elif ext == ".csv":
            matrix = csr_matrix(np.loadtxt(path, delimiter=","))
            log.info("Loaded .csv → CSR")

        elif ext == ".json":
            with open(path) as f:
                data = json.load(f)
            if all(k in data for k in ("row", "col", "data", "shape")):
                matrix = coo_matrix(
                    (data["data"], (data["row"], data["col"])),
                    shape=tuple(data["shape"])
                ).tocsr()
            else:
                matrix = csr_matrix(np.array(data))
            log.info("Loaded .json → CSR")

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    else:
        matrix = path_or_matrix
        if issparse(matrix):
            matrix = matrix.tocsr()
        elif isinstance(matrix, np.ndarray):
            matrix = csr_matrix(matrix)
        else:
            raise TypeError(f"Unsupported input type: {type(matrix)}")

    log.info(f"Matrix ready (shape={matrix.shape}, nnz={matrix.nnz})")
    return SafeCSR(matrix)


# ---- backward compatibility ---------------------------------

def adjacencyinfocheck(*args, **kwargs):
    """
    Backward-compatible alias for load_adjacency().
    """
    return load_adjacency(*args, **kwargs)
