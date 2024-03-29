
from .distance import distance, distance_dense
from .neighbors_util import compute_connectivities_umap
from .neighbors_batch import neighbors_batch
from .pcor import extract_pcor_info

def _neighbors_full(adata, use_rep, n_neighbors, min_std,
                    random_state, verbose, squared_correlation=False, pcor_inv=None):
    from umap.umap_ import nearest_neighbors
    si = adata.uns[use_rep]["bin_info"]
    metric_kwds = {"min_std": min_std,
                   "mids_x": si["mids_x"], "mids_y": si["mids_y"],
                   "mean_grid": si["mean"], "std_grid": si["std"],
                   "squared_correlation": squared_correlation,
                   "pcor_inv": pcor_inv}
    rep = adata.uns[use_rep]["rep"]
    X = adata.varm[rep]
    if 10 * n_neighbors > adata.shape[1]:
        knn_indices, knn_dists = distance_dense(X, n_neighbors=n_neighbors, **metric_kwds)
    else:
        knn_indices, knn_dists, _ = nearest_neighbors(X,
                                                      n_neighbors=n_neighbors,
                                                      metric=distance,
                                                      metric_kwds=metric_kwds,
                                                      angular=True,
                                                      random_state=random_state,
                                                      verbose=verbose)
    return knn_indices, knn_dists


def neighbors(adata, n_neighbors=5, key_added=None, use_rep="epiclust", min_std=0.001,
              random_state=0, verbose=False, set_op_mix_ratio=1.0, local_connectivity=1.0):
    """rip off of scipy nearest neighbors, but does not transpose anndata"""
    from sklearn.utils import check_random_state
    si = adata.uns[use_rep]["bin_info"]
    params = {"method": "umap",
              "random_state": random_state,
              "metric": "custom",
              "module": use_rep,
              "use_rep": adata.uns[use_rep]["rep"]}
    pcor_params = extract_pcor_info(adata, key=use_rep)
    if "batch_key" in adata.uns[use_rep].keys():
        knn_indices, knn_dists = neighbors_batch(adata,
                                                 use_rep=use_rep,
                                                 n_neighbors=n_neighbors,
                                                 min_std=min_std,
                                                 random_state=check_random_state(
                                                     random_state),
                                                 verbose=verbose,
                                                 squared_correlation=adata.uns[use_rep]["squared_correlation"],
                                                 pcor_inv=pcor_params.get("pcor_inv"))
    else:
        knn_indices, knn_dists = _neighbors_full(adata,
                                                 use_rep=use_rep,
                                                 n_neighbors=n_neighbors,
                                                 min_std=min_std,
                                                 random_state=check_random_state(
                                                     random_state),
                                                 verbose=verbose,
                                                 squared_correlation=adata.uns[use_rep]["squared_correlation"],
                                                 pcor_inv=pcor_params.get("pcor_inv"))
    distances, connectivities = compute_connectivities_umap(knn_indices, knn_dists,
                                                            knn_indices.shape[0],
                                                            knn_indices.shape[1],
                                                            set_op_mix_ratio=set_op_mix_ratio,
                                                            local_connectivity=local_connectivity)
    params["n_neighbors"] = knn_indices.shape[1]
    if key_added is None:
        key_added = "pow_%.2f" % adata.uns[use_rep]["power"]
    if "graphs" in adata.uns[use_rep].keys():
        adata.uns[use_rep]["graphs"] = sorted(list(set([key_added] + adata.uns[use_rep]["graphs"])))
    conns_key = key_added + "_connectivities"
    dists_key = key_added + "_distances"
    adata.uns[key_added] = {}
    neighbors_dict = adata.uns[key_added]
    neighbors_dict["connectivities_key"] = conns_key
    neighbors_dict["distances_key"] = dists_key
    neighbors_dict["params"] = params
    adata.varp[dists_key] = distances
    adata.varp[conns_key] = connectivities
