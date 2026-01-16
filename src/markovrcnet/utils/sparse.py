from scipy.sparse import csr_matrix

class SafeCSR(csr_matrix):
    def __repr__(self):
        return f"<SafeCSR shape={self.shape}, nnz={self.nnz}, dtype={self.dtype}>"

    __str__ = __repr__
