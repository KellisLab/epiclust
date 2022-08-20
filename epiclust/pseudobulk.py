
def pseudobulk(pbdf, adata, columns=["leiden", "Sample"], obsm=None, varm=None):
    """ALL pbdf columns are intersected when using .concat"""
    import numpy as np
    import pandas as pd
    import anndata
    import scipy.sparse
    I = np.intersect1d(pbdf.index.values, adata.obs.index.values)
    pbdf = pbdf.loc[I, :]
    cls_idx = pbdf.groupby(columns).ngroup()
    ucls, cls_idx, cls_inv = np.unique(cls_idx, return_index=True, return_inverse=True)
    adata_inv = adata.obs.index.get_indexer(I)
    S = scipy.sparse.csr_matrix((np.ones(len(cls_inv)),
                                 (cls_inv, adata_inv)), dtype=int,
                                shape=(len(ucls), adata.shape[0]))
    dtype = np.int
    if obsm in adata.obsm.keys() and varm in adata.varm.keys():
        X = S.dot(adata.obsm[obsm]).dot(adata.varm[varm].T)
        dtype = np.float32
    elif adata.isbacked:
        adata = adata.to_memory()
        X = S.dot(adata.X)
        dtype = adata.X.dtype
    else:
        X = S.dot(adata.X)
        dtype = adata.X.dtype
    obs = pd.DataFrame(index=ucls)
    for x in np.setdiff1d(pbdf.columns, col):
        for i, cls in enumerate(ucls):
            allval = pbdf[x].values[i == cls_inv]
            if not np.all(allval == allval[0]):
                break
        else:
            obs[x] = pbdf[x].values[cls_idx]
    return anndata.AnnData(X, dtype=dtype,
                           obs=obs,
                           var=adata.var)
