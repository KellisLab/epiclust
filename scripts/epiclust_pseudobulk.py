#!/usr/bin/env python3
from epiclust.pseudobulk import pseudobulk
from epiclust.gene_distance import df_to_pyranges, peak_names_to_var
from epiclust.gtf import annotate_ranges
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from sklearn.feature_extraction.text import TfidfTransformer
import sys, os, argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", nargs="+", metavar="KEY=VALUE")
    ap.add_argument("-t", "--table", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-c", "--columns", nargs="+", default=["leiden"])
    ap.add_argument("--tf-idf", nargs="+")
    ap.add_argument("--n-comps", default=100, type=int)
    ap.add_argument("--gtf-annotation", default="")
    args = vars(ap.parse_args())
    if args["input"] is None:
        print("No input!")
        sys.exit(1)
    data_list = {}
    data_keys = []
    for kv in args["input"]:
        S = [s.strip() for s in kv.split("=")]
        if len(S) == 2:
            data_list[S[0]] = anndata.read(S[1], backed="r")
            data_keys.append(S[0])
        else:
            print(S, "is not a valid key-value pair")
            sys.exit(1)
    df = pd.read_csv(args["table"], sep="\t", index_col=0)
    L = {}
    for k in data_keys:
        print("Pseudobulking", k)
        adata = pseudobulk(df, data_list[k], columns=args["columns"])
        if k in args["tf_idf"]:
            print("Running TF-IDF for", k)
            adata.X = TfidfTransformer().fit_transform(adata.X).astype(np.float32)
        sc.pp.normalize_total(adata, target_sum=10000)
        L[k] = adata
    adata = anndata.concat(L, axis=1, label="feature_types", merge="same")
    adata.var_names_make_unique("-ft-")
    sc.pp.log1p(adata)
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=[])
    sc.pp.pca(adata, n_comps=min(np.min(adata.shape)-1, args["n_comps"]))
    if os.path.exists(args["gtf_annotation"]):
        peaks = df_to_pyranges(peak_names_to_var(adata.var.index.values), chrom="seqname", left="start", right="end")
        ft = adata.var["feature_types"].values.astype(str)
        del adata.var["feature_types"]
        adata.var["feature_types"] = ft
        adata.var.loc[peaks.df["name"].values, "feature_types"] = annotate_ranges(peaks, args["gtf_annotation"])
    adata.write_h5ad(args["output"], compression="gzip")
