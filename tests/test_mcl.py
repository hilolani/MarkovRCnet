def test_mcl_with_dataset():
    from markovrcnet.datasets import load_all_adjmats
    from markovrcnet.mcl import mclprocess
    mats = load_all_adjmats()
    result = mclprocess(mats["karateclub"], 20)
    print(result)
