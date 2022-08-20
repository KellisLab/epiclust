
def pseudobulk(pbdf, adata, columns=["leiden", "Sample"], obsm=None, varm=None):
    """TODO ensure ALL pbdf columns are met"""
    import numpy as np
    import pandas as pd
    import anndata
    import scipy.sparse
    I = np.intersect1d(pbdf.values, adata.obs.index.values)
    pbdf = pbdf.loc[I, :]
    cls_idx = pbdf.groupby(columns).ngroup()
    ucls, cls_idx, cls_inv = np.unique(cls_idx, return_index=True, return_inverse=True)
    adata_inv = np.ravel(np.where(np.isin(adata.obs.index.values, I)))
    S = scipy.sparse.csr_matrix((np.ones(len(cls_inv)),
                                 (cls_inv, adata_inv)), dtype=int,
                                shape=(len(ucls), adata.shape[1]))
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
        for x in np.setdiff1d(integ_df.columns, col):
        for i, cls in enumerate(ucls):
            allval = integ_df[x].values[i == cls_inv]
            if not np.all(allval == allval[0]):
                break
        else:
            obs[x] = integ_df[x].values[cls_idx]
    return anndata.AnnData(X, dtype=dtype,
                           obs=obs,
                           var=adata.var)
