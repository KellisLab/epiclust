from .utils import *

import numpy as np
from tqdm.auto import tqdm
import dask
def calc_perbin_stats(rep, bin_assign, margin_of_error=0.05, z=2, n_bins_sample=2):
        rep = rep.astype(np.double)
        uniq = np.unique(bin_assign)
        nbin = len(uniq)
        ss_numer = z * z * 0.25 / (margin_of_error * margin_of_error) ### sample size = ss_numer / (1 + ss_numer/n)
        counts = np.zeros((nbin, nbin), dtype=int)
        means = np.zeros((nbin, nbin))
        stds = np.zeros((nbin, nbin))
        n_bins_sample = max(1, n_bins_sample)
        out = []
        for i in range(nbin):
                for j in range(i, nbin):
                        row_indices = np.where(np.abs(uniq[i] - bin_assign) < n_bins_sample)[0]
                        col_indices = np.where(np.abs(uniq[j] - bin_assign) < n_bins_sample)[0]
                        ss_n = np.sqrt(len(row_indices)*len(col_indices)) ### take harmonic mean which will balance the # of actual comparisons
                        ss = int(np.ceil(ss_numer / (1 + ss_numer/ss_n)))
                        row_indices = np.random.choice(row_indices, min(len(row_indices), int(ss)), replace=False)
                        col_indices = np.random.choice(col_indices, min(len(col_indices), int(ss)), replace=False)
                        ret = calc_stats_per_bin(rep, row_indices, col_indices, out_row=i, out_col=j)
                        out.append(ret)
        # for x in dask.compute(out)[0]:
        for x in out:
                counts[x["row"], x["col"]] = x["counts"]
                counts[x["col"], x["row"]] = x["counts"]
                means[x["row"], x["col"]] = x["mean"]
                means[x["col"], x["row"]] = x["mean"]
                stds[x["row"], x["col"]] = x["std"]
                stds[x["col"], x["row"]] = x["std"]
        return {"counts": counts,
                "mean": means,
                "std": stds}

def create_bins_quantile(margin, nbins=50):
        spt = np.array_split(np.sort(margin), nbins)
        edges = [np.min(x) for x in spt] + [np.max(margin)]
        indices = np.digitize(margin, edges[:-1])
        return indices, np.asarray(edges)
