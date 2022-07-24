
import numpy as np

def calc_stats_per_bin(X_adj, row_indices, col_indices, out_row, out_col, eps=1e-16):
    cor = X_adj[row_indices, :] @ X_adj[col_indices, :].T
    data = np.clip(cor.astype(np.float64), a_min=-1+eps, a_max=1-eps)
    data = data[~np.equal.outer(row_indices, col_indices)]
    data = np.arctanh(data)
    return {"counts": len(data),
            "mean": np.mean(data),
            "std": np.std(data),
            "row": out_row,
            "col": out_col}
