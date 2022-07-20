

def _gather_varp(adata, graph_name_list):
    L = {}
    for g in set(graph_name_list):
        if g not in adata.uns.keys():
            continue
        if "connectivities_key" not in adata.uns[g]:
            continue
        conn = adata.uns[g]["connectivities_key"]
        if conn not in adata.varp.keys():
            continue
        L[g] = conn
    return L

def _filter_var(adata, conn, z=None, pct=0.0):
    """Recall that distances are computed as np.exp(-z) to turn correlations into distances"""
    import numpy as np
    if z is None:
        z = -np.log(adata.varp[conn].data).mean()
    max_distance = np.exp(-z)
    nn = (adata.varp[conn] > 0).sum(1)
    M = nn - (adata.varp[conn] > max_distance).sum(1)
    ### what percentage of nearest neighbors for each .var are close enough?
    I, _ = np.where(M > (nn.max() * pct))
    return I

def filter_var(adata, graph_name_list, z=None, pct=0.0):
    from functools import reduce
    import numpy as np
    G = _gather_varp(adata, graph_name_list)
    I = [_filter_var(adata, conn, z, pct) for conn in G]
    return adata.var.index.values[reduce(np.union1d, I)]

def infomap(adata, graph_name_list, key_added="infomap", prefix="M", **kwargs):
    from infomap import Infomap
    import pandas as pd
    im = Infomap(**kwargs)
    im.add_nodes(range(adata.shape[1]))
    for layer, conn in enumerate(_gather_varp(adata, graph_name_list).values()):
        G = adata.varp[conn].tocoo()
        for i in range(len(G.data)):
            im.add_multilayer_intra_link(layer_id=layer,
                                         source_node_id=int(G.row[i]),
                                         target_node_id=int(G.col[i]),
                                         weight=G.data[i])
        del G
    im.run()
    adata.var[key_added] = pd.Categorical(["%s%d" % (prefix, node.module_id) for node in im.nodes])

def leiden(adata, graph_name_list, key_added="leiden", resolution=1., prefix="M", **kwargs):
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
    adata.var[key_added] = pd.Categorical(["%s%d" % (prefix, x+1) for x in memb])

def embedding(adata, graph_name, prefix="X_", **kwargs):
    from sklearn.manifold import spectral_embedding
    conn = _gather_varp([graph_name]).values()[0]
    se = spectral_embedding(adata.varp[conn], **kwargs)
    adata.varm["%s%s" % (prefix, graph_name)] = se

def select_clusters(adata, clust_name, graph_name, key_added="selected", power=-0.5):
    from sklearn.metrics import silhouette_score
    import numpy as np
    import scipy.sparse
    conn = list(_gather_varp(adata, [graph_name]).values())[0]
    uc, cinv, cnt = np.unique(adata.var[clust_name], return_inverse=True, return_counts=True)
    S = scipy.sparse.csr_matrix((cnt[cinv]**power,
                                 (np.arange(len(cinv)),
                                  cinv)))
    CS = adata.varp[conn].dot(S)
    top = np.ravel(CS[np.arange(len(cinv)), cinv])
    bot = np.ravel(CS.sum(1))
