from .utils import *

import numpy as np
from tqdm.auto import tqdm
def calc_perbin_stats(rep, bin_assign, margin_of_error=0.01, z=2):
        uniq = np.unique(bin_assign)
        nbin = len(uniq)
        ss_numer = z * z * 0.25 / (margin_of_error * margin_of_error) ### ss = ss_numer / (1 + numer/n)
        counts = np.zeros((nbin, nbin), dtype=int)
        means = np.zeros((nbin, nbin))
        stds = np.zeros((nbin, nbin))
        order = []
        for i in range(nbin):
                for j in range(i, nbin):
                        order.append((i,j))
        for i, j in tqdm(order):
                row_indices = np.where(uniq[i] == bin_assign)[0]
                col_indices = np.where(uniq[j] == bin_assign)[0]
                ss_n = np.sqrt(len(row_indices)*len(col_indices))
                ss = int(np.ceil(ss_numer / (1 + ss_numer/ss_n)))
                row_indices = np.random.choice(row_indices, min(len(row_indices), int(ss)), replace=False)
                col_indices = np.random.choice(col_indices, min(len(col_indices), int(ss)), replace=False)
                bstats = calc_stats_per_bin(rep, row_indices, col_indices)
                counts[j,i] = counts[i,j] = bstats["counts"]
                means[j,i] = means[i,j] = bstats["mean"]
                stds[j,i] = stds[i,j] = bstats["std"]
        return {"counts": counts,
                "mean": means,
                "std": stds}

def create_bins_quantile(margin, nbins=50):
        spt = np.array_split(np.sort(margin), nbins)
        edges = [np.min(x) for x in spt] + [np.max(margin)]
        indices = np.digitize(margin, edges[:-1])
        return indices, np.asarray(edges)
