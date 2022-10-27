#!/usr/bin/env python3

import os
import argparse

def process(h5ad, output, power, covariates=[], batch=None, margin="log1p_total_counts", n_neighbors=5, resolution=2, max_comm_size=None, min_comm_size=2, min_cells=3, infomap=False):
    import scanpy as sc
    import epiclust as ec
    import pandas as pd
    adata = sc.read(h5ad, backed="r")
    for p in power:
        print("Fitting data, power=%.2f" % p)
        ec.fit(adata, power=p,
               batch=batch, covariates=covariates,
               margin=margin)
        print("Finding %d nearest neighbors, power=%.2f" % (n_neighbors, p))
        ec.neighbors(adata, n_neighbors=n_neighbors)
    print("Filtering .var")
    ec.filter_var(adata, ["pow_%.2f" % p for p in power], min_cells=min_cells)
    if max_comm_size is not None and max_comm_size > 0:
        print("Finding clusters with resolution=%.2f and max_comm_size=%d" % (resolution, max_comm_size))
        ec.leiden(adata, ["pow_%.2f" % x for x in power],
                  resolution=resolution,
                  min_comm_size=min_comm_size,
                  max_comm_size=max_comm_size)
    else:
        print("Finding clusters with resolution=%.2f" % resolution)
        ec.leiden(adata, ["pow_%.2f" % x for x in power],
                  resolution=resolution,
                  min_comm_size=min_comm_size)
    if infomap:
        print("Running InfoMap")
        ec.infomap(adata, ["pow_%.2f" % x for x in power],
                   min_comm_size=min_comm_size,
                   preferred_number_of_modules=len(pd.unique(adata.var["leiden"])))
    print("Computing UMAP")
    ec.umap(adata, ["pow_%.2f" % x for x in power])
    print("Writing data")
    adata.write_h5ad(output, compression="gzip")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, dest="h5ad")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-p", "--power", type=float, nargs="+", default=[0., 0.25, 0.5, 0.75, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    ap.add_argument("--covariates", nargs="+")
    ap.add_argument("--batch")
    ap.add_argument("--margin", default="log1p_total_counts")
    ap.add_argument("--min-cells", type=int, default=3)
    ap.add_argument("--n-neighbors", type=int, default=5)
    ap.add_argument("-r", "--resolution", type=float, default=2.)
    ap.add_argument("--max-comm-size", type=int, default=None)
    ap.add_argument("--min-comm-size", type=int, default=3)
    ap.add_argument("--run-infomap", dest="infomap", action="store_true")
    ap.add_argument("--no-run-infomap", dest="infomap", action="store_false")
    ap.set_defaults(infomap=False)
    args = vars(ap.parse_args())
    process(**args)
