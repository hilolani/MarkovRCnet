def test_mif_broadcast_with_dataset():
    from markovrcnet.datasets import load_all_adjmats
    from markovrcnet.mif import MiF
    mats = load_all_adjmats()
    result0 = MiF_broadcast(mats["karateclub"], 3)
    result1 = MiF_broadcast(mats["karateclub"], 3, loop = 1)
    return result0, result1
