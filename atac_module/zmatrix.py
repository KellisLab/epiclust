import numpy as np
from tqdm.auto import tqdm
from functools import partial
import multiprocessing
from .utils import cov_to_cor_z_along
from .h5writer import H5Writer
import h5py
import pandas as pd
import dask
import dask.dataframe as dd

def fill_matrix(margin, X_adj, bin_assign, spline_table, z, writer, correct=None):
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
                if correct is not None:
                        new_cor_mat = correct(cor_mat, row_indices, col_indices)
                        ### tone down outliers from partial correlation
                        cor_mat = np.mean((cor_mat, new_cor_mat), axis=0)
                del s_tables
                cor_mat[np.equal.outer(row_indices, col_indices)] = -np.inf
                grow, gcol = np.where(cor_mat >= z)
                if len(grow) > 0 and len(gcol) == len(grow):
                        writer.add(row=row_indices[grow],
                                   col=col_indices[gcol],
                                   data=cor_mat[grow,gcol])
        return 0

def finalize(mat, row, col, cutoff):
        i, j = np.where(mat >= cutoff)
        if len(i) > 0:
                R = row[i]
                C = col[j]
                ind = R != C
                data = mat[i[ind], j[ind]]
                return pd.DataFrame({"row": R[ind],
                                     "col": C[ind],
                                     "data": data})
        else:
                return pd.DataFrame({"row": [],
                                     "col": [],
                                     "data": []})

def fill_matrix_dask(margin, X_adj, bin_assign, spline_table, z, writer, correct=None):
        """bin assign could be any assignment, since spline_table takes in margin itself.
        so, bin_assign could be e.g. chromosome positioning"""
        out = []
        uniq = np.unique(bin_assign)
        nbin = len(uniq)
        res = []
        for i in range(nbin):
                for j in range(i, nbin):
                        row_indices = np.where(uniq[i] == bin_assign)[0]
                        col_indices = np.where(uniq[j] == bin_assign)[0]
                        cor_mat = dask.delayed(cov_to_cor_z_along)(X_adj, row_indices=row_indices, col_indices=col_indices)
                        s_mean = dask.delayed(spline_table["mean"])(row_indices, col_indices)
                        s_std = dask.delayed(spline_table["std"])(row_indices, col_indices)
                        cor_mat = (cor_mat - s_mean) / s_std
                        if correct is not None:
                                new_cor_mat = dask.delayed(correct)(cor_mat, row_indices, col_indices)
                                ### tone down outliers from partial correlation
                        else:
                                new_cor_mat = cor_mat
                        done = dask.delayed(finalize)(new_cor_mat, row_indices, col_indices, z)
                        res.append(done)
        df = dd.from_delayed(res).persist() ### TODO: transform into Client.persist() ??
        dd.concat([df, df.rename(columns={"row": "col", "col": "row"})]).drop_duplicates(["row","col"]).to_hdf(writer["output"], "matrix")
        with h5py.File(writer["output"], "r+") as W:
                W["names"] = writer["names"]
