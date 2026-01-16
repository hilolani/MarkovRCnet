import os
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

from markovrcnet.utils.logging import resolve_logger
from markovrcnet.utils.sparse import SafeCSR



def save_safe_csr_to_mtx(safecsrmatrix, path: str, logger=None):
    log = logger or resolve_logger(None, "io")

    if hasattr(safecsrmatrix, "_csr"):
        safecsrmatrix = safecsrmatrix._csr

    if not isinstance(safecsrmatrix, csr_matrix):
        raise TypeError(f"Expected csr_matrix or SafeCSR, got {type(safecsrmatrix).__name__}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    mmwrite(path, safecsrmatrix)
    log.info(f"Saved CSR matrix to {path}")
