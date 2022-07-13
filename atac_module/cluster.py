
import infomap
import igraph

def infomap(adata, graph_name_list, key_added="infomap", **kwargs):
    im = infomap.Infomap(**kwargs)
    im.add_nodes(range(adata.shape[1]))
    for i, gname in enumerate(graph_name_list):
        G = adata.varp[gname].tocoo()
        for x in range(len(G.data)):
            im.add_multilayer_intra_link(i, G.row[x], G.col[x], weight=G.data[x])
    im.run()
    ### todo

def leiden(adata, graph_name_list, key_added="leiden", resolution=1., **kwargs):
    import scanpy as sc
    import leidenalg
    G = []
    for g in graph_name_list:
        if g not in adata.uns.keys() or "connectivities_key" not in adata.uns[g]:
            continue
        conn = adata.uns[g]["connectivities_key"]
        if conn not in adata.varp.keys():
            continue
        G.append(sc._utils.get_igraph_from_adjacency(adata.varp[conn]))
    memb, _ = leidenalg.find_partition_multiplex(G, leidenalg.RBConfigurationVertexPartition,
                                                 resolution_parameter=resolution, **kwargs)
    adata.var[key_added] = memb
