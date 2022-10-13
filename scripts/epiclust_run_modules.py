#!/usr/bin/env python3

import os
import argparse

def process(h5ad, output, power, covariates=[], batch=None, margin="log1p_total_counts", n_neighbors=25):
    import scanpy as sc
    import epiclust as ec
    adata = sc.read(h5ad, backed="r")
    for p in power:
        print("Fitting data, power = ", p)
        ec.fit(adata, power=p,
               batch=batch, covariates=covariates,
               margin=margin)
        print("Finding neighbors, power =", p)
        ec.neighbors(adata, n_neighbors=n_neighbors)
    print("Writing data")
    adata.write_h5ad(output, compression="gzip")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, dest="h5ad")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-p", "--power", type=float, nargs="+", default=[0., 0.25, 0.5, 0.75, 1.])
    ap.add_argument("--covariates", nargs="+")
    ap.add_argument("--batch")
    ap.add_argument("--margin", default="log1p_total_counts")
    ap.add_argument("--n-neighbors", type=int, default=25)
    args = vars(ap.parse_args())
    process(**args)
