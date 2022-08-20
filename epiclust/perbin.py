from .utils import calc_stats_per_bin

import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

def calc_perbin_stats(rep, bin_assign_row, bin_assign_col, where_row=None, where_col=None,
                      margin_of_error=0.05, z=2, n_bins_sample=1, blur=1, pcor_varm=None, pcor_inv=None):
        r_uniq = np.unique(bin_assign_row)
        c_uniq = np.unique(bin_assign_col)
        ss_numer = z * z * 0.25 / (margin_of_error * margin_of_error) ### sample size = ss_numer / (1 + ss_numer/n)
        counts = np.zeros((len(r_uniq), len(c_uniq)), dtype=int)
        means = np.zeros((len(r_uniq), len(c_uniq)))
        stds = np.zeros((len(r_uniq), len(c_uniq)))
        n_bins_sample = max(1, n_bins_sample)
        out = []
        for i in range(len(r_uniq)):
                for j in range(len(c_uniq)): ### not symmetric
                        row_indices = np.where(np.abs(r_uniq[i] - bin_assign_row) < n_bins_sample)[0]
                        col_indices = np.where(np.abs(c_uniq[j] - bin_assign_col) < n_bins_sample)[0]
                        if where_row is not None:
                                row_indices = where_row[row_indices]
                        if where_col is not None:
                                col_indices = where_col[col_indices]
                        ss_n = np.sqrt(len(row_indices)*len(col_indices)) ### take harmonic mean which will balance the # of actual comparisons
                        ss = int(np.ceil(ss_numer / (1 + ss_numer/ss_n)))
                        row_indices = np.random.choice(row_indices, min(len(row_indices), int(ss)), replace=False)
                        col_indices = np.random.choice(col_indices, min(len(col_indices), int(ss)), replace=False)
                        ret = calc_stats_per_bin(rep, row_indices, col_indices, out_row=i, out_col=j, pcor_inv=pcor_inv, pcor_varm=pcor_varm)
                        out.append(ret)
        for x in out:
                counts[x["row"], x["col"]] = x["counts"]
                means[x["row"], x["col"]] = x["mean"]
                stds[x["row"], x["col"]] = x["std"]
        means = gaussian_filter(means, blur)
        stds = gaussian_filter(stds, blur)
        return {"counts": counts,
                "mean": means,
                "std": stds}

def create_bins_quantile(margin, nbins=50):
        spt = np.array_split(np.sort(margin), nbins)
        edges = [np.min(x) for x in spt] + [np.max(margin)]
        edges = np.unique(edges) ### removes duplicate counts
        indices = np.digitize(margin, edges[:-1])
        return indices, np.asarray(edges)
