
import numpy as np
from .pcor import pcor_adjust

def calc_stats_per_bin(X_adj, row_indices, col_indices, out_row, out_col, eps=1e-16, pcor_varm=None, pcor_inv=None, squared_correlation=False):
    cor = X_adj[row_indices, :] @ X_adj[col_indices, :].T
    RI, CI = np.where(~np.equal.outer(row_indices, col_indices))
    data = cor[RI, CI].astype(np.float64)
    if pcor_varm is not None and pcor_inv is not None:
        data = pcor_adjust(cor=data,
                           row=row_indices[RI],
                           col=col_indices[CI],
                           varm=pcor_varm,
                           inv=pcor_inv)
    if squared_correlation:
        data *= data
    data = np.clip(data.astype(np.float64), a_min=-1+eps, a_max=1-eps)
    data = np.arctanh(data)
    return {"counts": len(data),
            "mean": np.mean(data),
            "std": np.std(data),
            "row": out_row,
            "col": out_col}
