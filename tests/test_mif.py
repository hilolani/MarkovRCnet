def test_mif_with_dataset():
    from markovrcnet.datasets import load_all_adjmats
    from markovrcnet.mif import MiF
    mats = load_all_adjmats()
    result = MiF(mats["karateclub"], 4, 32, 0.5, 3)
    print(result)
