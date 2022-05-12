import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import scipy.sparse
from functools import partial
import multiprocessing
from .utils import cov_to_cor_z_along
from .h5writer import H5Writer
def fill_matrix(margin, X_adj, bin_assign, spline_table, z, writer):
        """bin assign could be any assignment, since spline_table takes in margin itself.
        so, bin_assign could be e.g. chromosome positioning"""
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
                s_tables = {sname: spl(np.median(margin[row_indices]),
                                       np.median(margin[col_indices])) for sname, spl in spline_table.items()}
                cor_mat = cor_mat - s_tables["mean"]
                cor_mat = cor_mat / s_tables["std"]
                del s_tables
                cor_mat[np.equal.outer(row_indices, col_indices)] = -np.inf
                grow, gcol = np.where(cor_mat >= z)
                if len(grow) > 0 and len(gcol) == len(grow):
                        writer.add(row=row_indices[grow],
                                   col=col_indices[gcol],
                                   data=cor_mat[grow,gcol])
        return 0
