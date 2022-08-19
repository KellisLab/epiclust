#!/usr/bin/env python3
import epiclust.gene_estimation as ge
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--atac", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--linking", default=None)
    ap.add_argument("--filter-peaks", default="selected")
    ap.add_argument("--top-gene", default="closest")
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--obsm", default="")
    ap.add_argument("--varm", default="")
    ap.add_argument("--max-distance", type=int, default=1000000)
    ap.add_argument("--promoter-distance", type=int, default=5000)
    ap.add_argument("--target-sum", type=int, default=10000)
    ap.add_argument("--annotation", nargs="+")
    args = vars(ap.parse_args())
    if args["linking"] is not None:
        linking = pd.read_csv(args["linking"], sep="\t")
    else:
        linking = None
    gdata = ge.estimate_genes_linking(adata=anndata.read(args["atac"]),
                                      gtf=load_gtf(args["gtf"]),
                                      linking=linking,
                                      max_distance=args["max_distance"],
                                      promoter_distance=args["promoter_distance"],
                                      target_sum=args["target_sum"],
                                      obsm=args["obsm"],
                                      varm=args["varm"],
                                      top_gene=args["top_gene"])
    if args["annotation"]:
        ge.annotate(gdata, args["annotation"])
    gdata.write_h5ad(args["output"], compression="gzip")
