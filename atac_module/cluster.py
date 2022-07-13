
import igraph

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
    memb, _ = leidenalg.find_partition_multiplex(G, leidenalg.RBConfigurationVertexPartition,
                                                 resolution_parameter=resolution, **kwargs)
    adata.var[key_added] = pd.Categorical(["%s%d" % (prefix, x+1) for x in memb])
