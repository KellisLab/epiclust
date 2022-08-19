
import numpy as np
import numba
from .bispeu import bispeu_wrapped as bispeu
import numba
import logging
from numba import float64, types

@numba.njit
def distance(x, y, min_std=0.001, mids_x=None, mids_y=None, mean_grid=None, std_grid=None):
    """kwargs is .uns[key]"""
    dim = x.shape[0]
    result = np.zeros(1)
    mean = np.zeros(1)
    std = np.zeros(1)
    ix = (np.abs(x[0] - mids_x)).argmin()
    iy = (np.abs(y[0] - mids_y)).argmin()
    mean += mean_grid[ix, iy]
    std += std_grid[ix, iy]
    for i in range(1, dim):
        dot = x[i] * y[i]
        result += dot
    std[std < min_std] = min_std
    result[result < -1+1e-16] = -1+1e-16
    result[result > 1-1e-16] = 1-1e-16
    out = np.exp(-1 * (np.arctanh(result) - mean) / std)
    return(out[0])

def raw_correlation(adata, row, col, use_rep="X_scm", batch_size=10000):
    ncor = min(len(row), len(col))
    cor = np.zeros(ncor)
    X = adata.varm[use_rep]
    for begin in range(0, ncor, batch_size):
        end = min(begin + batch_size, ncor)
        out = np.multiply(X[row[begin:end], 1:],
                          X[col[begin:end], 1:]).sum(1)
        cor[begin:end] = out
    return np.arctanh(np.clip(cor, -1+1e-16, 1-1e-16))
