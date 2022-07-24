
from .distance import distance
from .spline import build_spline, build_ridge

def neighbors(adata, n_neighbors=15, key_added=None, use_rep="scm", min_std=0.001, random_state=0, verbose=False):
    """rip off of scipy nearest neighbors, but does not transpose anndata"""
    from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set
    from sklearn.utils import check_random_state
    import scipy.sparse
    import scanpy as sc
    si = adata.uns[use_rep]["spline_info"]
    metric_kwds = {"min_std": min_std, "mids": si["mids"],
                   "mean_grid": si["mean"], "std_grid": si["std"]}
    rep = adata.uns[use_rep]["rep"]
    X = adata.varm[rep]
    knn_indices, knn_dists, _ = nearest_neighbors(X,
                                                  n_neighbors=n_neighbors,
                                                  metric=distance,
                                                  metric_kwds=metric_kwds,
                                                  angular=True, ### correlation is angular-based
                                                  random_state=check_random_state(random_state),
                                                  verbose=verbose)
    X = scipy.sparse.lil_matrix((adata.shape[1], 1))
    connectivities, _, _ = fuzzy_simplicial_set(X,
                                                n_neighbors,
                                                None, None,
                                                knn_indices=knn_indices,
                                                knn_dists=knn_dists,
                                                set_op_mix_ratio=1.0,
                                                local_connectivity=1.0)
    distances = sc.neighbors._get_sparse_matrix_from_indices_distances_umap(
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        n_obs=adata.shape[1],
        n_neighbors=n_neighbors)
    if key_added is None:
        key_added = 'neighbors'
        conns_key = 'connectivities'
        dists_key = 'distances'
    else:
        conns_key = key_added + '_connectivities'
        dists_key = key_added + '_distances'
    adata.uns[key_added] = {}
    neighbors_dict = adata.uns[key_added]
    neighbors_dict['connectivities_key'] = conns_key
    neighbors_dict['distances_key'] = dists_key
    neighbors_dict['params'] = {'n_neighbors': n_neighbors, 'method': "umap"}
    neighbors_dict['params']['random_state'] = random_state
    neighbors_dict['params']['metric'] = "custom"
    neighbors_dict['params']['metric_kwds'] = metric_kwds
    neighbors_dict['params']['use_rep'] = rep
    adata.varp[dists_key] = distances
    adata.varp[conns_key] = connectivities.tocsr()
