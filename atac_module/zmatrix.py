import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import scipy.sparse

from .utils import cov_to_cor_z_along

def fill_matrix(margin, X_adj, bin_assign, spline_table, z):
        out = []
        uniq = np.unique(bin_assign)
        nbin = len(uniq)
        order = []
        for i in range(nbin):
                for j in range(i, nbin):
                        order.append((i,j))
        for i, j in tqdm(order):
                row_indices = np.where(uniq[i] == bin_assign)[0]
                col_indices = np.where(uniq[j] == bin_assign)[0]
                cor_mat = cov_to_cor_z_along(X_adj, row_indices=row_indices, col_indices=col_indices)
                s_tables = {sname: spl(margin[row_indices], margin[col_indices]) for sname, spl in spline_table.items()}
                cor_mat = cor_mat - s_tables["mean"]
                cor_mat = cor_mat / s_tables["std"]
                del s_tables
                cor_mat[np.equal.outer(row_indices, col_indices)] = -np.inf
                grow, gcol = np.where(cor_mat >= z)
                if len(grow) > 0 and len(gcol) == len(grow):
                        out.append(pd.DataFrame({"row": row_indices[grow], "col": col_indices[gcol], "data": cor_mat[grow, gcol]}))
        out = pd.concat(out)
        return scipy.sparse.csr_matrix((out["data"].values, (out["row"].values, out["col"].values)), shape=(X_adj.shape[0], X_adj.shape[0]))
