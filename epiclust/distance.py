
import numpy as np
import numba
from .bispeu import bispeu_wrapped as bispeu
import numba
import logging
from .pcor import pcor_adjust


@numba.njit
def distance(x, y, min_std=0.001, mids_x=None, mids_y=None,
             mean_grid=None, std_grid=None, squared_correlation=False, pcor_inv=None):
    """Keywords in distance() MUST BE put into dict IN ORDER for NNDescent"""
    dim = x.shape[0]
    if pcor_inv is not None:
        dim = dim - pcor_inv.shape[0]
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
    # apply pcor if applicable
    if pcor_inv is not None:
        npc = pcor_inv.shape[0]
        result = pcor_adjust(result,
                             row_varm=x[-npc:].reshape(1, -1).astype(np.float64),
                             col_varm=y[-npc:].reshape(1, -1).astype(np.float64),
                             inv=pcor_inv)
    if squared_correlation:
        result *= result
    result[result < -1 + 1e-16] = -1 + 1e-16
    result[result > 1 - 1e-16] = 1 - 1e-16
    out = np.exp(-1 * (np.arctanh(result) - mean) / std)
    return(out[0])

def distance_dense(X_epiclust, n_neighbors=-1, min_std=0.001, mids_x=None, mids_y=None,
                   mean_grid=None, std_grid=None, squared_correlation=False, pcor_inv=None):
    import numpy as np
    import scipy.sparse
    dim = X_epiclust.shape[1]
    if pcor_inv is not None:
        dim = dim - pcor_inv.shape[0]
    #result = np.zeros((X_epiclust.shape[0], X_epiclust.shape[0]), dtype=np.float32)
    ix = np.abs(X_epiclust[:, [0]] - mids_x[None, :]).argmin(1)
    iy = np.abs(X_epiclust[:, [0]] - mids_y[None, :]).argmin(1)
    #### TODO do block wise to ensure sparsity
    result = X_epiclust[:, 1:dim].astype(np.float64) @ X_epiclust[:, 1:dim].T.astype(np.float64)
    result = scipy.sparse.coo_matrix(result, dtype=np.float64)
    if pcor_inv is not None:
        npc = pcor_inv.shape[0]
        result.data = pcor_adjust(result.data,
                                  row_varm=X_epiclust[result.row, -npc:].astype(np.float64),
                                  col_varm=X_epiclust[result.col, -npc:].astype(np.float64),
                                  inv=pcor_inv)
    if squared_correlation:
        result.data *= result.data
    result.data[result.data < -1 + 1e-16] = -1 + 1e-16
    result.data[result.data > 1 - 1e-16] = 1 - 1e-16
    result = np.exp(-1 * (np.arctanh(result.todense()) - mean_grid[ix, :][:, iy]) / std_grid[ix, :][:, iy].clip(min_std, np.inf))
    indices = np.argsort(result, axis=1)
    ### now for indices; dists
    if n_neighbors < 0:
        n_neighbors = indices.shape[1]
    if n_neighbors < indices.shape[1]:
        indices = indices[:, np.arange(n_neighbors)]
    return indices, result[np.arange(result.shape[0])[:, None], indices].astype(np.float32)

@numba.njit
def correlation(X_rep, I_row, I_col, min_std=0.001, mids_x=None, mids_y=None, mean_grid=None,
                std_grid=None, pcor_inv=None, pcor_varm=None, squared_correlation=False):
    ncor = min(len(I_row), len(I_col))
    dim = X_rep.shape[1]
    result = np.zeros(ncor)
    mean = np.zeros(ncor)
    std = np.zeros(ncor)
    for i in range(ncor):
        ix = (np.abs(X_rep[I_row[i], 0] - mids_x)).argmin()
        iy = (np.abs(X_rep[I_col[i], 0] - mids_y)).argmin()
        mean[i] = mean_grid[ix, iy]
        std[i] = std_grid[ix, iy]
        for j in range(1, dim):
            result[i] += X_rep[I_row[i], j] * X_rep[I_col[i], j]
    std[std < min_std] = min_std
    # apply pcor if applicable
    if pcor_inv is not None and pcor_varm is not None:
        result = pcor_adjust(result, row=I_row, col=I_col,
                             varm=pcor_varm, inv=pcor_inv)
    if squared_correlation:
        result *= result
    result[result < -1 + 1e-16] = -1 + 1e-16
    result[result > 1 - 1e-16] = 1 - 1e-16
    return (np.arctanh(result) - mean) / std
