# cython: profile=True
import numpy as np
cimport numpy as np
cimport cython
import scipy.sparse
from tqdm.auto import tqdm

ctypedef fused DTYPE_t:
    float
    double

@cython.boundscheck(False)
@cython.wraparound(False)
def cov_to_cor_z_along(np.ndarray[DTYPE_t, ndim=2] X_adj,
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
def calc_stats_per_bin(np.ndarray[DTYPE_t, ndim=2] X_adj,
                       np.ndarray[Py_ssize_t, ndim=1] row_indices,
                       np.ndarray[Py_ssize_t, ndim=1] col_indices):
        data = cov_to_cor_z_along(X_adj, row_indices, col_indices)
        data = data[~np.equal.outer(row_indices, col_indices)]
        return {"counts": len(data),
                "mean": np.mean(data),
                "std": np.std(data)}

def isPD(X):
        try:
                _ = np.linalg.cholesky(X)
                return True
        except np.linalg.LinAlgError:
                return False


def nearPD(np.ndarray[DTYPE_t, ndim=2] A):
        B = (A + A.T)/2
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A2 = (B + H)/2
        A3 = (A2 + A2.T) / 2
        if isPD(A3):
                return A3
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not isPD(A3):
                min_eig = np.min(np.real(np.linalg.eigvals(A3)))
                A3 += I * (-min_eig * k**2 + spacing)
                k += 1
        return A3

@cython.boundscheck(False)
@cython.wraparound(False)
def _outer_cor(A, B):
        top = A @ B.T - A.shape[1] * np.multiply.outer(A.mean(1), B.mean(1))
        bottom = A.shape[1] * np.multiply.outer(np.std(A, axis=1),
                                                np.std(B, axis=1))
        quot = np.divide(top, bottom, out=np.zeros_like(top), where=bottom != 0)
        return np.clip(quot, a_min=-1, a_max=1)

@cython.boundscheck(False)
@cython.wraparound(False)
def outer_correlation_svd(np.ndarray[DTYPE_t, ndim=2] U,
                          np.ndarray[DTYPE_t, ndim=1] s,
                          np.ndarray[DTYPE_t, ndim=2] VT,
                          np.ndarray[DTYPE_t, ndim=2] B,
                          Py_ssize_t batch_size=1000):
        assert VT.shape[1] == B.shape[1]
        out = np.zeros((U.shape[0], B.shape[0]))
        t = tqdm(total=np.prod(out.shape))
        for a_begin in range(0, U.shape[0], batch_size):
                a_end = min(a_begin + batch_size, U.shape[0])
                Ma = U[a_begin:a_end, :] @ np.diag(s) @ VT
                Ma = np.asarray(Ma)
                for b_begin in range(0, B.shape[0], batch_size):
                        b_end = min(b_begin + batch_size, B.shape[0])
                        Mb = B[b_begin:b_end, :]
                        if(scipy.sparse.issparse(Mb)):
                                Mb = Mb.todense()
                        Mb = np.asarray(Mb)
                        out[a_begin:a_end, b_begin:b_end] = _outer_cor(Ma, Mb)
                        t.update((a_end-a_begin)*(b_end-b_begin))
        t.close()
        return out

def outer_correlation(np.ndarray[DTYPE_t, ndim=2] A,
                      np.ndarray[DTYPE_t, ndim=2] B,
                      Py_ssize_t batch_size=1000):
        assert A.shape[1] == B.shape[1]
        out = np.zeros((A.shape[0], B.shape[0]))
        for a_begin in range(0, A.shape[0], batch_size):
                a_end = min(a_begin + batch_size, A.shape[0])
                Ma = A[a_begin:a_end, :]
                if scipy.sparse.issparse(Ma):
                        Ma = Ma.todense()
                Ma = np.asarray(Ma)
                for b_begin in range(0, B.shape[0], batch_size):
                        b_end = min(b_begin + batch_size, B.shape[0])
                        Mb = B[b_begin:b_end, :]
                        if scipy.sparse.issparse(Mb):
                                Mb = Mb.todense()
                        Mb = np.asarray(Mb)
                        out[a_begin:a_end, b_begin:b_end] = _outer_cor(Ma, Mb)
        return out
