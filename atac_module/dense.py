from .spline import build_spline, spline_grid_ordered

def neighbors_dense(adata, use_rep="scm", spline_k=2, min_std=0.001, key_added="connectivities", n_neighbors=10):
    import numpy as np
    import scipy.sparse
    mean_spl = build_spline(adata, key=use_rep, spline="mean", k=spline_k)
    std_spl = build_spline(adata, key=use_rep, spline="std", k=spline_k)
    vrep = adata.uns[use_rep]["rep"]
    margin = adata.varm[vrep][:, 0]
    data = adata.varm[vrep][:, 1:]
    A = data @ data.T
    A = np.arctanh(A.clip(-1+1e-16, 1-1e-16))
    M = spline_grid_ordered(mean_spl,
                            margin,
                            margin)
    S = spline_grid_ordered(std_spl,
                            margin,
                            margin)
    A = (A - M) / S.clip(min_std, np.inf)
    del M, S
    ### get cutoff based on desired n_neighbors
    q1, q2 = np.triu_indices(A.shape[0], 1)
    cutoff = np.quantile(A[q1, q2], 1 - n_neighbors / A.shape[0])
    A[A < cutoff] = 0
    np.fill_diagonal(A, 0)
    A = scipy.sparse.csr_matrix(A)
    D = np.ravel(A.sum(0))
    D[D < cutoff] = 1
    D = scipy.sparse.diags(D**-0.5)
    adata.varp[key_added] = D.dot(A).dot(D).astype(np.float32)
