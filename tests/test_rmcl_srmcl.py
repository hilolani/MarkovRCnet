def test_rmcl_sr_mcl_with_dataset():
    from markovrcnet.datasets import load_all_adjmats
    import markovrcnet.mcl as mcl
    mats = load_all_adjmats()
    cluslist = mcl.mclprocess(mats["scalefree"], 20)
    result_srall = mcl.sr_mcl(cluslist, mats["scalefree"], coreinfoonly = False)
    return result_srall

