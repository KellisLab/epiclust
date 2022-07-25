

def get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors):
    """copy from scanpy"""
    import numpy as np
    import scipy.sparse
    rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)
    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            else:
                val = knn_dists[i, j]

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    result = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
    result.eliminate_zeros()
    return result.tocsr()

def compute_connectivities_umap(knn_indices, knn_dists,
                                n_obs, n_neighbors, set_op_mix_ratio=1.0,
                                local_connectivity=1.0):
    """copy from scanpy"""
    import scipy.sparse
    from umap.umap_ import fuzzy_simplicial_set
    X = scipy.sparse.lil_matrix((n_obs, 1))
    connectivities = fuzzy_simplicial_set(X, n_neighbors, None, None,
                                          knn_indices=knn_indices, knn_dists=knn_dists,
                                          set_op_mix_ratio=set_op_mix_ratio,
                                          local_connectivity=local_connectivity)
    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]
    distances = get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors)
    return distances, connectivities.tocsr()
