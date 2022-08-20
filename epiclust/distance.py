
import numpy as np
import numba
from .bispeu import bispeu_wrapped as bispeu
import numba
import logging
from .pcor import pcor_adjust

@numba.njit
def distance(x, y, min_std=0.001, mids_x=None, mids_y=None, mean_grid=None, std_grid=None, squared_correlation=False):
    """kwargs is .uns[key]"""
    dim = x.shape[0]
    result = np.zeros(1)
    mean = np.zeros(1)
    std = np.zeros(1)
    ix = np.argmin(np.abs(x[0] - mids_x))
    iy = np.argmin(np.abs(y[0] - mids_y))
    mean += mean_grid[ix, iy]
    std += std_grid[ix, iy]
    for i in range(1, dim):
        dot = x[i] * y[i]
        result += dot
    std[std < min_std] = min_std
    if squared_correlation:
        result *= result
    result[result < -1+1e-16] = -1+1e-16
    result[result > 1-1e-16] = 1-1e-16
    out = np.exp(-1 * (np.arctanh(result) - mean) / std)
    return(out[0])

@numba.njit
def correlation(X_rep, I_row, I_col, min_std=0.001, mids_x=None, mids_y=None, mean_grid=None, std_grid=None, pcor_inv=None, pcor_varm=None, squared_correlation=False):
    ncor = min(len(I_row), len(I_col))
    dim = X_rep.shape[1]
    result = np.zeros(ncor)
    mean = np.zeros(ncor)
    std = np.zeros(ncor)
    for i in range(ncor):
        ix = np.argmin(np.abs(X_rep[I_row[i], 0] - mids_x))
        iy = np.argmin(np.abs(X_rep[I_col[i], 0] - mids_y))
        mean[i] = mean_grid[ix, iy]
        std[i] = std_grid[ix, iy]
        for j in range(1, dim):
            result[i] += X_rep[I_row[i], j] * X_rep[I_col[i], j]
    std[std < min_std] = min_std
    ### apply pcor if applicable
    if pcor_inv is not None and pcor_varm is not None:
        result = pcor_adjust(result, row=I_row, col=I_col, varm=pcor_varm, inv=pcor_inv)
    if squared_correlation:
        result *= result
    result[result < -1+1e-16] = -1+1e-16
    result[result > 1-1e-16] = 1-1e-16
    return (np.arctanh(result) - mean) / std
