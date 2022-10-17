
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

def _filter_var(G, z=2, pct=0.0):
    """Recall that distances are computed as np.exp(-z) to turn correlations into distances
"""
    import numpy as np
    import scipy.stats
    percentile = scipy.stats.norm.cdf(z)
    percentile = min(percentile, 1 - percentile)
    max_distance = np.quantile(G.data, percentile)
    nn = (G > 0).sum(1)
    M = nn - (G > max_distance).sum(1)
    # what percentage of nearest neighbors for each .var are close enough?
    row, _ = np.where(M > (nn.max() * pct))
    return row

def _gather_batch_indices(adata, use_rep="epiclust", selected="selected"):
    """ minus 1 for nonselected indices"""
    import numpy as np
    if "batch_key" in adata.uns[use_rep].keys():
        ub, binv = np.unique(
            adata.var[adata.uns[use_rep]["batch_key"]], return_inverse=True)
    else:
        binv = np.zeros(adata.shape[1], dtype=int)
        ub = ["all"]
    if selected is not None:
        if selected in adata.var.columns and adata.var[selected].dtype == np.dtype('bool'):
            idx = np.ravel(np.where(adata.var[selected]))
            binv[idx] = -1
        else:
            print("Warning: No boolean column named \"%s\" in .var" % selected)
    return ub, binv

def _gather_graphs(adata, graph_name_list, split_batch=True, use_rep="epiclust", selected="selected", graph="connectivities_key"):
    import scanpy as sc
    import scipy.sparse
    import numpy as np
    G = []
    ubatch, batches = _gather_batch_indices(adata, use_rep=use_rep, selected=selected)
    for conn in _gather_varp(adata, graph_name_list, graph=graph).values():
        V = adata.varp[conn]
        if split_batch and len(ubatch) > 1:
            for i, _ in enumerate(ubatch):
                ### Enumerating over ubatch removes -1s from batches since starts at 0
                S = scipy.sparse.diags(batches == i, dtype=int)
                VS = V.dot(S)
                for j, _ in enumerate(ubatch):
                    S = scipy.sparse.diags(batches == j, dtype=int)
                    G.append(S.dot(VS))
        else:
            ### Use "batches" to remove non-selected indices
            S = scipy.sparse.diags(batches >= 0, dtype=int)
            G.append(S.dot(V.dot(S)))
    if not G:
        raise RuntimeError("No graphs present")
    return [g for g in G if len(g.data) > 0]

def filter_var(adata, graph_name_list, z=2, pct=0.0, use_rep="epiclust", key_added="selected", split_batch=True):
    from functools import reduce
    import numpy as np
    G = _gather_graphs(adata, graph_name_list, graph="distances_key",
                       use_rep=use_rep, split_batch=split_batch, selected=None)
    adata.var[key_added] = False
    for g in G:
        I = _filter_var(g, z=z, pct=pct)
        adata.var.loc[adata.var.index.values[I], key_added] = True

def infomap(adata, graph_name_list, key_added="G_infomap", split_batch=True, use_rep="epiclust", selected="selected", min_comm_size=2, min_centrality=0, **kwargs):
    from infomap import Infomap
    import pandas as pd
    import numpy as np
    import scipy.sparse
    im = Infomap(**kwargs)
    GV = _gather_graphs(adata, graph_name_list,
                        split_batch=split_batch,
                        use_rep=use_rep,
                        selected=selected)
    for layer, G in enumerate(GV):
        G = G.tocoo()
        for i in range(len(G.data)):
            im.add_multilayer_intra_link(layer_id=layer,
                                         source_node_id=int(G.row[i]),
                                         target_node_id=int(G.col[i]),
                                         weight=G.data[i])
    im.run()
    assign = []
    ### Get all module assignments per node
    for node in im.get_tree(depth_level=1, states=True):
        if node.is_leaf and node.modular_centrality > min_centrality:
            assign.append((node.module_id, node.node_id, node.modular_centrality))
    df = pd.DataFrame(assign, columns=["module_id", "node_id", "modular_centrality"])
    um, minv, mcnt = np.unique(df["module_id"].values, return_inverse=True, return_counts=True)
    good = np.ravel(np.where(mcnt >= min_comm_size))
    df = df.iloc[np.isin(minv, good), :]
    _, df["module_id"] = np.unique(df["module_id"].values, return_inverse=True)
    adata.varm[key_added] = scipy.sparse.csr_matrix(
        (df["modular_centrality"].values,
         (df["node_id"].values,
          df["module_id"].values)),
        shape=(adata.shape[1], len(pd.unique(df["module_id"]))),
        dtype=np.float32)

def combine_graphs(adata, graph_name_list, selected="selected", use_rep="epiclust"):
    import numpy as np
    G = None
    for V in _gather_graphs(adata, graph_name_list,
                            split_batch=False, use_rep=use_rep,
                            selected=selected):
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
           use_rep="epiclust", selected="selected", min_comm_size=2,
           resolution=1., prefix="M", **kwargs):
    import scanpy as sc
    import leidenalg
    import pandas as pd
    import numpy as np
    G = [sc._utils.get_igraph_from_adjacency(g) for g in _gather_graphs(adata,
                                                                        graph_name_list,
                                                                        split_batch=split_batch,
                                                                        use_rep=use_rep,
                                                                        selected=selected)]
    clust, _ = leidenalg.find_partition_multiplex(G, leidenalg.RBConfigurationVertexPartition,
                                                  resolution_parameter=resolution, **kwargs)
    clust = np.asarray(clust)
    uclust, cinv, ccnt = np.unique(clust, return_counts=True, return_inverse=True)
    bad_clust = np.ravel(np.where(ccnt < min_comm_size))
    clust[np.isin(cinv, bad_clust)] = -1
    adata.var[key_added] = pd.Categorical(["%s%d" % (prefix, x + 1) if x >= 0 else None for x in clust ])
