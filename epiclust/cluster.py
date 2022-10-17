from functools import reduce


def _gather_varp(adata, graph_name_list, graph="connectivities_key"):
    L = {}
    if type(graph_name_list) == "str":
        graph_name_list = [graph_name_list]
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


def _gather_batch_indices(adata, use_rep="epiclust"):
    if "batch_key" in adata.uns[use_rep].keys():
        ub, binv = np.unique(
            adata.var[adata.uns[use_rep]["batch_key"]], return_inverse=True)
    else:
        binv = np.zeros(adata.shape[1], dtype=int)
        ub = ["all"]
    return ub, binv

def _gather_graphs(adata, graph_name_list, split_batch=True, use_rep="epiclust"):
    import scanpy as sc
    import scipy.sparse
    import numpy as np
    G = []
    ubatch, batches = _gather_batch_indices(adata, use_rep=use_rep)
    for conn in _gather_varp(adata, graph_name_list).values():
        V = adata.varp[conn]
        if split_batch and len(ubatch) > 1:
            for i, _ in enumerate(ubatch):
                S = scipy.sparse.diags(batches == i, dtype=int)
                VS = V.dot(S)
                for j, _ in enumerate(ubatch):
                    S = scipy.sparse.diags(batches == j, dtype=int)
                    VS = S.dot(VS)
                    G.append(VS)
        else:
            G.append(V)
    if not G:
        raise RuntimeError("No graphs present")
    return [g for g in G if len(g.data) > 0]

def infomap(adata, graph_name_list, key_added="infomap", split_batch=True, use_rep="epiclust", prefix="M", **kwargs):
    from infomap import Infomap
    import pandas as pd
    from tqdm.auto import tqdm
    import numpy as np
    im = Infomap(**kwargs)
    GV = _gather_graphs(adata, graph_name_list, split_batch=split_batch, use_rep=use_rep)
    for layer, G in enumerate(GV):
        G = G.tocoo()
        for i in tqdm(np.arange(len(G.data))):
            im.add_multilayer_intra_link(layer_id=layer,
                                         source_node_id=int(G.row[i]),
                                         target_node_id=int(G.col[i]),
                                         weight=G.data[i])
    im.run()
    clust = np.ones(adata.shape[1], dtype=int) * -1
    for node in im.tree:
        if node.is_leaf:
            clust[node.node_id] = node.module_id
    adata.var[key_added] = pd.Categorical(["M%d" % x if x >= 0 else None for x in clust ])

def combine_graphs(adata, graph_name_list):
    import numpy as np
    G = None
    for conn in _gather_varp(adata, graph_name_list).values():
        V = adata.varp[conn]
        if G is None:
            G = V
        else:
            G = sparse_maximum(G, V)
    return G

def top_features_per_group(adata, graph_name_list, groupby="leiden", n=10):
    import numpy as np
    import scanpy as sc
    import pandas as pd
    G = sc._utils.get_igraph_from_adjacency(combine_graphs(adata, graph_name_list))
    G.vs["groupby"] = adata.var[groupby].values
    G.vs["name"] = adata.var.index.values
    tbl = {}
    for cls in pd.unique(adata.var[groupby]):
        sg = G.subgraph(G.vs.select(groupby=cls))
        pr = sg.pagerank()
        idx = np.argsort(pr)[::-1][range(min(n, len(pr)))]
        tbl[cls] = np.asarray(sg.vs["name"])[idx]
    return tbl

def leiden(adata, graph_name_list, key_added="leiden", split_batch=True,
           use_rep="epiclust",
           resolution=1., prefix="M", **kwargs):
    import scanpy as sc
    import leidenalg
    import pandas as pd
    G = [sc._utils.get_igraph_from_adjacency(g) for g in _gather_graphs(adata, graph_name_list, split_batch=split_batch, use_rep=use_rep)]
    memb, _ = leidenalg.find_partition_multiplex(G, leidenalg.RBConfigurationVertexPartition,
                                                 resolution_parameter=resolution, **kwargs)
    adata.var[key_added] = pd.Categorical(
        ["%s%d" % (prefix, x + 1) for x in memb])


def embedding(adata, graph_name, prefix="X_", **kwargs):
    from sklearn.manifold import spectral_embedding
    conn = _gather_varp(adata, [graph_name]).values()[0]
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
