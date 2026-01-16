import os
from sklearn.utils import Bunch

def load_adjmats():
    base_path = os.path.join(os.path.dirname(__file__), "..", "data")
    base_path = os.path.abspath(base_path)

    return Bunch(
        erdosReny=os.path.join(base_path, "ErdosReny.mtx"),
        gadget=os.path.join(base_path, "gadget.mtx"),
        heterophilly=os.path.join(base_path, "heterophilly.mtx"),
        homophilly=os.path.join(base_path, "homophilly.mtx"),
        karateclub=os.path.join(base_path, "karateclub.mtx"),
        scalefree=os.path.join(base_path, "scalefree.mtx"),
        eat=os.path.join(base_path, "eat.mtx"),
        DESCR="Toy adjacency matrices in Matrix Market format"
    )


# aliases
load_mif = load_adjmats
load_mcl = load_adjmats
