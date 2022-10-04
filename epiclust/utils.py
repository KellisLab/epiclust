
import numpy as np
from .pcor import pcor_adjust


def calc_stats_per_bin(X_adj, row_indices, col_indices, out_row, out_col,
                       eps=1e-16, n_pcs=-1, pcor_inv=None, squared_correlation=False):
    cor = X_adj[row_indices, :n_pcs] @ X_adj[col_indices, :n_pcs].T
    RI, CI = np.where(~np.equal.outer(row_indices, col_indices))
    data = cor[RI, CI].astype(np.float64)
    if pcor_inv is not None:
        ninv = pcor_inv.shape[0]
        data = pcor_adjust(cor=data,
                           row_varm=X_adj[row_indices[RI], -ninv:].astype(np.float64),
                           col_varm=X_adj[col_indices[CI], -ninv:].astype(np.float64),
                           inv=pcor_inv)
    if squared_correlation:
        data *= data
    data = np.clip(data.astype(np.float64), a_min=-1 + eps, a_max=1 - eps)
    data = np.arctanh(data)
    return {"counts": len(data),
            "mean": np.mean(data),
            "std": np.std(data),
            "row": out_row,
            "col": out_col}
