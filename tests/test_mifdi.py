def test_mifdi_with_dataset():
    from markovrcnet.datasets import load_all_adjmats
    import markovrcnet.mif as mif
    mats = load_all_adjmats()
    result0 = mif.MiFDI(mats["karateclub"], dangn = 0)
    result1 = mif.MiFDI(mats["karateclub"], dangn = 0, loop = 1)
    return result0, result1
