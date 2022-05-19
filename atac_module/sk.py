import numpy as np
import scipy.sparse
import sparse
import dask.array as da
import dask

def _np_as_sparse(arr):
        return da.asarray(sparse.COO.from_scipy_sparse(scipy.sparse.diags(arr)))

def sinkhorn_knopp_dask(A, rowSums, colSums, epsilon=1e-3, max_iter=1000):
        if scipy.sparse.issparse(A):
                ### Use compatible sparse matrix library for dask
                A = sparse.COO.from_scipy_sparse(A)
        A = da.from_array(A)
        r = da.asarray(rowSums)
        c = da.asarray(colSums)
        x = da.ones(A.shape[0])
        y = da.ones(A.shape[1])
        D1 = _np_as_sparse(np.ones(A.shape[0]))
        D2 = _np_as_sparse(np.ones(A.shape[1]))
        max_r = da.absolute(D1.dot(A).dot(y) - r)
        max_c = da.absolute(D2.dot(A.T).dot(x) - c)
        max_max = dask.compute(da.hstack((max_r, max_c)).max())[0]
        iteration = 0
        while iteration < max_iter and max_max > epsilon:
                print("iter:", iteration, "max difference:", max_max)
                y_new = c / A.T.dot(x)
                y = dask.compute(y_new)[0]
                x_new = r / A.dot(y)
                x = dask.compute(x_new)[0]
                D1 = _np_as_sparse(x)
                D2 = _np_as_sparse(y)
                x = da.asarray(x)
                y = da.asarray(y)
                max_r = da.absolute(D1.dot(A).dot(y) - r)
                max_c = da.absolute(D2.dot(A.T).dot(x) - c)
                max_max = dask.compute(da.hstack((max_r, max_c)).max())[0]
                iteration += 1
        return dask.compute(x)[0], dask.compute(y)[0]

def sinkhorn_knopp(A, rowSums, colSums, epsilon=1e-5, max_iter=1000):
        x = np.ones(A.shape[0])
        y = np.ones(A.shape[1])
        diff_r = scipy.sparse.diags(x).dot(A).dot(y) - rowSums
        diff_c = scipy.sparse.diags(y).dot(A.T).dot(x) - colSums
        max_rc = max(np.abs(diff_r).max(), np.abs(diff_c).max())
        iteration = 0
        while iteration < max_iter and max_rc > epsilon:
                print(iteration, ":", max_rc)
                y = colSums / A.T.dot(x)
                x = rowSums / A.dot(y)
                diff_r = scipy.sparse.diags(x).dot(A).dot(y) - rowSums
                diff_c = scipy.sparse.diags(y).dot(A.T).dot(x) - colSums
                max_rc = max(np.abs(diff_r).max(), np.abs(diff_c).max())
                iteration += 1
        return x, y
