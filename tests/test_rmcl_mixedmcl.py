def test_rmcl_mixedmcl_with_dataset():
    from markovrcnet.datasets import load_all_adjmats
    import markovrcnet.mcl as mcl
    mats = load_all_adjmats()
    cluslist = mcl.mclprocess(mats["scalefree"], 20)
    mixed_result =  mcl.mixed_rmcl(cluslist, mats["scalefree"], threspruning = 3.0, branching = False)
    mixed_result_dict = mcl.mcllist_to_mcldict(mixed_result)
    return mixed_result_dict
