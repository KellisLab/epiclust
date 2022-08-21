
def sinkhorn_knopp(A, rowSums=None, colSums=None,
                   epsilon=1e-5, max_iter=1000, verbose=True):
    import numpy as np
    import scipy.sparse
    if rowSums is None:
        rowSums = A.shape[1] * np.ones(A.shape[0], dtype=np.int64)
    if colSums is None:
        colSums = A.shape[0] * np.ones(A.shape[1], dtype=np.int64)
    x = np.ones(A.shape[0], dtype=np.int64)
    y = np.ones(A.shape[1], dtype=np.int64)
    diff_r = scipy.sparse.diags(x, dtype=np.int64).dot(A).dot(y) - rowSums
    diff_c = scipy.sparse.diags(y, dtype=np.int64).dot(A.T).dot(x) - colSums
    max_rc = max(np.abs(diff_r).max(), np.abs(diff_c).max())
    iteration = 0
    while iteration < max_iter and max_rc > epsilon:
        if verbose:
            print("SK", iteration, ":", max_rc)
        y = colSums / A.T.dot(x)
        x = rowSums / A.dot(y)
        diff_r = scipy.sparse.diags(x).dot(A).dot(y) - rowSums
        diff_c = scipy.sparse.diags(y).dot(A.T).dot(x) - colSums
        max_rc = max(np.abs(diff_r).max(), np.abs(diff_c).max())
        iteration += 1
    return x, y


def biwhiten(adata, key="sk", n_comps=None,
             verbose=True, max_iter=1000, epsilon=1e-5):
    import numpy as np
    import scipy.sparse
    import scanpy as sc
    x, y = sinkhorn_knopp(adata.X, verbose=verbose,
                          max_iter=max_iter, epsilon=epsilon)
    adata.obs["sk"] = x
    adata.var["sk"] = y
    X = adata.X.copy()
    adata.X = scipy.sparse.diags(np.sqrt(x)).dot(X).dot(
        scipy.sparse.diags(np.sqrt(y))).astype(np.float32)
    cutoff = np.sqrt(adata.shape).sum()
    if n_comps is None:
        n_comps = (1 + np.sqrt(np.max(adata.shape) / np.min(adata.shape)))**2
    n_comps = min(np.min(adata.shape) - 1, n_comps)
    if verbose:
        print("Running SVD for", n_comps, "components")
    sc.pp.pca(adata, n_comps=n_comps, zero_center=False)
    adata.X = X
