"""
Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

from .distance import distance
import warnings

def neighbors_batch(adata, use_rep, n_neighbors, min_std,
                    random_state, verbose, squared_correlation=False, pcor_inv=None):
    """This function is derived from get_graph in bbknn (https://github.com/Teichlab/bbknn)"""
    import numpy as np
    from pynndescent import NNDescent
    si = adata.uns[use_rep]["bin_info"]
    batches = adata.uns[use_rep]["batches"]
    B = adata.var[adata.uns[use_rep]["batch_key"]]
    rep = adata.varm[adata.uns[use_rep]["rep"]]
    knn_dists = np.zeros((rep.shape[0], n_neighbors * len(batches)))
    knn_indices = np.copy(knn_dists).astype(int)
    for x, xbatch in enumerate(batches):
        xmask = np.ravel(np.where(B.values == xbatch))
        for y, ybatch in enumerate(batches):
            metric_kwds = {"min_std": min_std}
            if x <= y:
                key = "%s %s" % (xbatch, ybatch)
                metric_kwds["mids_x"] = si[key]["mids_x"]
                metric_kwds["mids_y"] = si[key]["mids_y"]
                metric_kwds["mean_grid"] = si[key]["mean"]
                metric_kwds["std_grid"] = si[key]["std"]
            else:
                key = "%s %s" % (ybatch, xbatch)
                metric_kwds["mids_x"] = si[key]["mids_y"]
                metric_kwds["mids_y"] = si[key]["mids_x"]
                metric_kwds["mean_grid"] = si[key]["mean"].T
                metric_kwds["std_grid"] = si[key]["std"].T
            metric_kwds["squared_correlation"] = squared_correlation
            metric_kwds["pcor_inv"] = pcor_inv
            ymask = np.ravel(np.where(B.values == ybatch))
            tree = NNDescent(rep[xmask, :], metric=distance, metric_kwds=metric_kwds,
                             n_jobs=-1, n_neighbors=n_neighbors,
                             max_candidates=60,
                             random_state=random_state, verbose=verbose)
            with warnings.catch_warnings():  # warning for csr structure change in pynnd
                warnings.simplefilter("ignore")
                tree.prepare()
            I, D = tree.query(rep[ymask, :], k=n_neighbors)
            col_range = np.arange(x * n_neighbors, (x + 1) * n_neighbors)
            knn_indices[ymask[:, None], col_range[None, :]] = xmask[I]
            knn_dists[ymask[:, None], col_range[None, :]] = D
    newidx = np.argsort(knn_dists, axis=1)
    knn_indices = knn_indices[np.arange(np.shape(knn_indices)[0])[:, np.newaxis],
                              newidx]
    knn_distances = knn_dists[np.arange(np.shape(knn_dists)[0])[:, np.newaxis],
                              newidx]
    return knn_indices, knn_distances
