#!/usr/bin/env python3

import os
import argparse

def process(h5ad, output, power=0., covariates=[], batch=None, margin="log1p_total_counts", batch_size=1000, juicer=False):
    import csv
    import scanpy as sc
    import epiclust as ec
    import pandas as pd
    import scipy.stats
    adata = sc.read(h5ad, backed="r")
    print("Fitting data, power=%.2f" % power)
    ec.fit(adata, power=power,
           batch=batch, covariates=covariates,
           margin=margin)
    print("Computing dense matrix")
    if juicer:
        ec.dense(adata, output, transform=ec.hic.juicer_hic, batch_size=batch_size)
    else:
        ec.dense(adata, output, batch_size=batch_size)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, dest="h5ad")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-p", "--power", type=float, default=0.)
    ap.add_argument("--covariates", nargs="+")
    ap.add_argument("--batch")
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--margin", default="log1p_total_counts")
    ap.add_argument("--juicer", dest="juicer", action="store_true")
    ap.set_defaults(juicer=False)
    args = vars(ap.parse_args())
    process(**args)
