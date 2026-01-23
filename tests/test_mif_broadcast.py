def test_mif_broadcast_with_dataset():
    from markovrcnet.datasets import load_all_adjmats
    import markovrcnet.mif as mif
    mats = load_all_adjmats()
    result0 = mif.MiF_broadcast(mats["karateclub"], 3)
    result1 = mif.MiF_broadcast(mats["karateclub"], 3, loop = 1)
    return result0, result1
