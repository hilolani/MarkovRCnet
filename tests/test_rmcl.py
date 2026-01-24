def test_rmcl_with_dataset():
    from markovrcnet.datasets import load_all_adjmats
    import markovrcnet.mcl as mcl
    mats = load_all_adjmats()
    cluslist = mcl.mclprocess(mats["scalefree"], 20)
    result_branching = mcl.rmcl_basic(cluslist, mats["scalefree"], threspruning=1, reverse_process=False) 
    return result_branching

