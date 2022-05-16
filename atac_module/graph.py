#!/usr/bin/env python3
import numpy as np
import igraph
import scipy.sparse
import leidenalg
import pandas as pd
import h5py
import argparse

def create_graph(adj, vertex_names, min_degree=3, min_size=2, a_max=18.7):
        if np.max(adj) <= 1:
                print("warning: should use transform for graph")
        adj[adj > a_max] = 0
        degree = np.ravel((adj > 0).sum(1))
        adj = adj[degree >= min_degree,:][:,degree >= min_degree]

        vertex_names = vertex_names[degree >= min_degree]
        adj = scipy.sparse.coo_matrix(adj)
        g = igraph.Graph(list(tuple(zip(adj.row, adj.col))),
                         vertex_attrs={"name": vertex_names})
        g.simplify(combine_edges="max")
        g.es["weight"] = adj.data
        ### prune components
        memb = g.clusters().membership
        u, c = np.unique(memb, return_counts=True)
        comp = set(u[c < min_size])
        to_delete = np.asarray([i for i, x in enumerate(memb) if x in comp])
        g.delete_vertices(to_delete)
        ### prune nodes
        todel = [i for i, x in enumerate(g.degree()) if x <= 0]
        g.delete_vertices(todel)
        return g


def index_of(values, idx):
        """search for values inside idx"""
        sort_idx = np.argsort(idx)
        values_idx = sort_idx[np.searchsorted(idx, values, sorter=sort_idx)]
        assert np.all(idx[values_idx] == values)
        return values_idx

def cluster_graphs(graph_list, **kwargs):
        """do things the naive way first."""
        nodes = list()
        for g in graph_list:
                nodes += list(g.vs["name"])
        nodes, counts = np.unique(nodes,return_counts=True)
        nodes = nodes[counts == np.max(counts)]
        for i in range(len(graph_list)):
                g = graph_list[i]
                g.delete_vertices([i for i, name in enumerate(g.vs["name"]) if name not in nodes])
                node_idx = index_of(g.vs["name"], nodes) ### xfer graph vertex name -> global name index
                gdat = [(node_idx[e.source], node_idx[e.target]) for e in g.es] ## so that graph index -> global index
                g1 = igraph.Graph(gdat, vertex_attrs={"name": nodes}) ### can assign global node list
                del gdat
                g1.simplify(combine_edges="max")
                g1.es["weight"] = list(g.es["weight"])
                graph_list[i] = g1
                del g
        memb, _ = leidenalg.find_partition_multiplex(graph_list, leidenalg.RBConfigurationVertexPartition, n_iterations=-1, seed=0, **kwargs)
        return pd.DataFrame({"gene": nodes, "memb": memb})

if __name__ == "__main__":
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--input", required=True, nargs="+")
        ap.add_argument("-o", "--output", default="/dev/stdout")
        ap.add_argument("--max-comm-size", default=0, type=int)
        ap.add_argument("-r", "--resolution", default=1., type=float)
        args = vars(ap.parse_args())
        gl = []
        for fname in args["input"]:
                with h5py.File(fname, "r") as F:
                        print("Reading", fname)
                        var_names = np.asarray([x.decode() for x in F["name"][:]])
                        S = scipy.sparse.csr_matrix((F["data"][:],
                                                     (F["row"][:],
                                                      F["col"][:])),
                                                    shape=(len(var_names),
                                                           len(var_names)))
                        g = create_graph(S, var_names)
                        del S
                        gl.append(g)
        cargs = {"resolution_parameter": args["resolution"]}
        if args["max_comm_size"] > 0:
                cargs["max_comm_size"] = args["max_comm_size"]
        df = cluster_graphs(gl, **cargs)
        df.to_csv(args["output"], sep="\t", header=False, index=False)
