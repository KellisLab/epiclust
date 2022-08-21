from functools import reduce


def _gather_varp(adata, graph_name_list, graph="connectivities_key"):
    L = {}
    for g in set(graph_name_list):
        if g not in adata.uns.keys():
            continue
        if graph not in adata.uns[g]:
            continue
        conn = adata.uns[g][graph]
        if conn not in adata.varp.keys():
            continue
        L[g] = conn
    return L


def sparse_maximum(A, B):
    import numpy as np
    diff = A - B
    diff.data = np.where(diff.data < 0, 1, 0)
    return A - A.multiply(diff) + B.multiply(diff)


def _filter_var(adata, conn, z=2, pct=0.0, row_indices=None, col_indices=None):
    """Recall that distances are computed as np.exp(-z) to turn correlations into distances
"""
    import numpy as np
    import scipy.stats
    percentile = scipy.stats.norm.cdf(z)
    percentile = min(percentile, 1 - percentile)
    X = adata.varp[conn]
    if row_indices is not None:
        X = X[row_indices, :]
    if col_indices is not None:
        X = X[:, col_indices]
    max_distance = np.quantile(X.data, percentile)
    nn = (X > 0).sum(1)
    M = nn - (X > max_distance).sum(1)
    # what percentage of nearest neighbors for each .var are close enough?
    I, _ = np.where(M > (nn.max() * pct))
    if row_indices is not None:
        return row_indices[I]
    else:
        return I


def filter_var(adata, graph_name_list, z=2, pct=0.0, use_rep="epiclust"):
    """TODO: detect batches"""
    from functools import reduce
    import numpy as np
    G = _gather_varp(adata, graph_name_list, graph="distances_key")
    I = []
    if "batch_key" in adata.uns[use_rep].keys():
        ub, binv = np.unique(
            adata.var[adata.uns[use_rep]["batch_key"]], return_inverse=True)
        for i, _ in enumerate(ub):
            R = np.ravel(np.where(binv == i))
            for j, _ in enumerate(ub):
                C = np.ravel(np.where(binv == j))
                I += [_filter_var(adata, conn, z, pct,
                                  row_indices=R,
                                  col_indices=C) for conn in G.values()]
        # add row-indices and col_indices for each batch pair and add to I list
    else:
        I = [_filter_var(adata, conn, z, pct) for conn in G.values()]
    return adata.var.index.values[reduce(np.union1d, I)]


def infomap(adata, graph_name_list, key_added="infomap", prefix="M", **kwargs):
    from infomap import Infomap
    import pandas as pd
    im = Infomap(**kwargs)
    im.add_nodes(range(adata.shape[1]))
    for layer, conn in enumerate(
            _gather_varp(adata, graph_name_list).values()):
        G = adata.varp[conn].tocoo()
        for i in range(len(G.data)):
            im.add_multilayer_intra_link(layer_id=layer,
                                         source_node_id=int(G.row[i]),
                                         target_node_id=int(G.col[i]),
                                         weight=G.data[i])
        del G
    im.run()
    adata.var[key_added] = pd.Categorical(
        ["%s%d" % (prefix, node.module_id) for node in im.nodes])


def leiden(adata, graph_name_list, key_added="leiden",
           resolution=1., prefix="M", **kwargs):
    import scanpy as sc
    import leidenalg
    import pandas as pd
    G = []
    for conn in _gather_varp(adata, graph_name_list).values():
        V = adata.varp[conn]
        G.append(sc._utils.get_igraph_from_adjacency(V))
    if not G:
        raise RuntimeError("No graphs present")
    memb, _ = leidenalg.find_partition_multiplex(G, leidenalg.RBConfigurationVertexPartition,
                                                 resolution_parameter=resolution, **kwargs)
    adata.var[key_added] = pd.Categorical(
        ["%s%d" % (prefix, x + 1) for x in memb])


def embedding(adata, graph_name, prefix="X_", **kwargs):
    from sklearn.manifold import spectral_embedding
    conn = _gather_varp([graph_name]).values()[0]
    se = spectral_embedding(adata.varp[conn], **kwargs)
    adata.varm["%s%s" % (prefix, graph_name)] = se


def select_clusters(adata, clust_name, graph_name,
                    key_added="selected", power=-0.5):
    from sklearn.metrics import silhouette_score
    import numpy as np
    import scipy.sparse
    conn = list(_gather_varp(adata, [graph_name]).values())[0]
    uc, cinv, cnt = np.unique(
        adata.var[clust_name], return_inverse=True, return_counts=True)
    S = scipy.sparse.csr_matrix((cnt[cinv]**power,
                                 (np.arange(len(cinv)),
                                  cinv)))
    CS = adata.varp[conn].dot(S)
    top = np.ravel(CS[np.arange(len(cinv)), cinv])
    bot = np.ravel(CS.sum(1))
