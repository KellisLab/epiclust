#!/usr/bin/env python3
import anndata
from epiclust.gtf import annotate_ranges
from epiclust.gene_distance import df_to_pyranges, peak_names_to_var

import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-g", "--gtf", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-c", "--column", default="feature_types")
    args = vars(ap.parse_args())
    adata = anndata.read(args["input"])
    pvar = peak_names_to_var(adata.var.index.values)
    annot = annotate_ranges(df_to_pyranges(pvar, chrom="seqname", left="start", right="end"), gtf_file=args["gtf"])
    if args["column"] not in adata.var.columns:
        adata.var[args["column"]] = "Non-peak"
    else:
        oldcol = adata.var[args["column"]].astype(str).values
        del adata.var[args["column"]]
        adata.var[args["column"]] = oldcol
    adata.var.loc[pvar.index.values, args["column"]] = annot
    adata.write_h5ad(args["output"], compression="gzip")
