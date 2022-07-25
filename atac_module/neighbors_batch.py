
from .distance import distance
import warnings
def neighbors_batch(adata, use_rep, n_neighbors, min_std, random_state, verbose):
    """similar to get_graph in bbknn"""
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
            if x < y:
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
            ymask = np.ravel(np.where(B.values == ybatch))
            tree = NNDescent(rep[xmask, :], metric=distance, metric_kwds=metric_kwds,
                             n_jobs=-1, n_neighbors=n_neighbors,
                             max_candidates=60,
                             random_state=random_state, verbose=verbose)
            with warnings.catch_warnings(): #### warning for csr structure change in pynnd
                warnings.simplefilter("ignore")
                tree.prepare()
            I, D = tree.query(rep[ymask, :], k=n_neighbors)
            col_range = np.arange(x * n_neighbors, (x+1) * n_neighbors)
            knn_indices[ymask[:, None], col_range[None, :]] = xmask[I]
            knn_dists[ymask[:, None], col_range[None, :]] = D
    newidx = np.argsort(knn_dists, axis=1)
    knn_indices = knn_indices[np.arange(np.shape(knn_indices)[0])[:, np.newaxis],
                              newidx]
    knn_distances = knn_dists[np.arange(np.shape(knn_dists)[0])[:, np.newaxis],
                              newidx]
    return knn_indices, knn_distances
