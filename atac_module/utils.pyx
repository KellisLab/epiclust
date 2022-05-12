# cython: profile=True
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cov_to_cor_z_along(np.ndarray[double, ndim=2] X_adj,
                       np.ndarray[Py_ssize_t, ndim=1] row_indices,
                       np.ndarray[Py_ssize_t, ndim=1] col_indices,
                       eps=1e-16):
        ### Use row and col indices to compute correlation matrix from X_adj
        ### which should be std-normalized (assert np.all(np.linalg.norm(X_adj, axis=1, ord=2) == 1.))
        ### then clips at extremes to avoid errors
        if row_indices is None:
                row_indices = np.arange(X_adj.shape[0])
        if col_indices is None:
                col_indices = np.arange(X_adj.shape[0])
        cor = X_adj[row_indices,:] @ X_adj[col_indices,:].T
        return np.arctanh(np.clip(cor, a_min=-1+eps, a_max=1-eps))

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_stats_per_bin(np.ndarray[double, ndim=2] X_adj,
                       np.ndarray[Py_ssize_t, ndim=1] row_indices,
                       np.ndarray[Py_ssize_t, ndim=1] col_indices):
        data = cov_to_cor_z_along(X_adj, row_indices, col_indices)
        data = data[~np.equal.outer(row_indices, col_indices)]
        return {"counts": len(data),
                "mean": np.mean(data),
                "std": np.std(data)}




# def outer_correlation_svd():
#         return 0

# def outer_correlation(np.ndarray[double, ndim=2] A,
#                       np.ndarray[double, ndim=2] B,
#                       Py_ssize_t batch_size=1000):
#         assert A.shape[1] == B.shape[1]

# def nearPD(mat):

# def adjust_coo(mat, row_indices, col_indices, batch_size=5000, lfcor, rfcor, ainv):
