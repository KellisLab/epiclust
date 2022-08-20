#!/usr/bin/env python3
import scanpy as sc
import anndata
import numpy as np
import pandas as pd
import scipy.sparse
from epiclust.gene_distance import peak_names_to_var
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-b", "--bed", default="")
    ap.add_argument("-c", "--column", default="leiden")
    ap.add_argument("-f", "--feature-types", nargs="+", default=["Peaks"])
    ap.add_argument("-g", "--groupby", default="leiden")
    ap.add_argument("--min-size", type=int, default=10)
    args = vars(ap.parse_args())
    adata = sc.read(args["input"], backed="r")
    adata = adata[:, adata.var["feature_types"].isin(args["feature_types"])].to_memory()
    uc, cnt = np.unique(adata.var["leiden"].values, return_counts=True)
    adata = adata[:, adata.var["leiden"].isin(uc[cnt >= args["min_size"]])].copy()
    if args["bed"]:
        V = peak_names_to_var(adata.var.index.values)
        V["leiden"] = adata.var.loc[V.index.values, "leiden"]
        V.loc[:, ["seqname", "start", "end", "leiden"]].to_csv(args["bed"], sep="\t", index=False, header=False)
    uc, cinv = np.unique(adata.var["leiden"].values, return_inverse=True)
    S = scipy.sparse.csr_matrix((np.ones(adata.shape[1]),
                                 (np.arange(adata.shape[1]),
                                  cinv)),
                                dtype=np.uint64)
    X = adata.raw[:, adata.var.index.values].X
    cdata = anndata.AnnData(np.asarray(X.dot(S).todense()),
                            obs=adata.obs,
                            var=pd.DataFrame(index=uc),
                            dtype=np.uint64)
    del X, S
    sc.pp.calculate_qc_metrics(cdata, inplace=True, percent_top=[])
    cdata.layers["raw"] = cdata.X.copy()
    sc.pp.normalize_total(cdata, target_sum=10000)
    sc.pp.log1p(cdata)
    cdata.obsm["X_pca"] = adata.obsm["X_pca"]
    sc.pp.neighbors(cdata)
    sc.tl.leiden(cdata)
    sc.tl.umap(cdata)
    sc.tl.rank_genes_groups(cdata, groupby=args["groupby"], method="wilcoxon", use_raw=False)
    cdata.write_h5ad(args["output"], compression="gzip")
