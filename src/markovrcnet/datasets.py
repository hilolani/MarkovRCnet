from importlib.resources import files
from sklearn.utils import Bunch

from .io import load_adjacency


# ---- new API -------------------------------------------------

def load_adjmats(name: str):
    """
    Load a single adjacency matrix by name and return it as SafeCSR.
    """
    path = files("markovrcnet.data") / f"{name}.mtx"
    A = load_adjacency(path)
    return Bunch(name=name, adjacency=A)


# ---- legacy API (compatibility) ------------------------------

def load_all_adjmats():
    """
    Legacy-style loader: return paths to all bundled datasets.
    """
    base = files("markovrcnet.data")
    return Bunch(
        erdosReny=str(base / "ErdosReny.mtx"),
        gadget=str(base / "gadget.mtx"),
        heterophilly=str(base / "heterophilly.mtx"),
        homophilly=str(base / "homophilly.mtx"),
        karateclub=str(base / "karateclub.mtx"),
        scalefree=str(base / "scalefree.mtx"),
        DESCR="Toy adjacency matrices in Matrix Market format",
    )


# aliases
load_mif = load_all_adjmats
load_mcl = load_all_adjmats

