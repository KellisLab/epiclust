#!/usr/bin/env python3
from .gtf import load_gtf
from .gene_distance import peak_names_to_var, distance_weight


def estimate_genes_linking(adata, gtf, linking=None, obsm="NMF_H", varm="NMF_W", top_gene=None,
                           max_distance=1e6,
                           promoter_distance=5000,
                           filter_peaks="selected",
                           target_sum=10000):
    import anndata
    import scipy.sparse
    import scanpy as sc
    import numpy as np
    if filter_peaks in adata.var.columns:
        print("Filtering peaks to \"%s\"" % filter_peaks)
        PN = peak_names_to_var(
            adata.var.loc[adata.var[filter_peaks], :].index.values)
    else:
        PN = peak_names_to_var(adata.var.index.values)
    print("Calculating distances")
    dw = distance_weight(enh=PN,
                         gene=gtf,
                         max_distance=max_distance,
                         margin=promoter_distance,
                         closest_gene=top_gene == "closest")
    DWM = scipy.sparse.csr_matrix((dw["weight"].values,
                                   (adata.var.index.get_indexer(dw["peak"].values),
                                    gtf.index.get_indexer(dw["gene"].values))),
                                  shape=(adata.shape[1], gtf.shape[0]))
    DWM = DWM.dot(scipy.sparse.diags(gtf["gene_length_score"].values))
    if linking is not None:
        if top_gene == "linking":
            linking = linking.sort_values(
                "prob", ascending=False).drop_duplicates("peak")
        LM = scipy.sparse.csr_matrix((linking["prob"].values,
                                      (adata.var.index.get_indexer(linking["peak"].values),
                                       gtf.index.get_indexer(linking["gene"].values))),
                                     shape=(adata.shape[1], gtf.shape[0]))
        print("Multiplying links times distances")
        DWM = DWM.multiply(LM)
        del LM
    print("Multiplying weights through ATAC peaks")
    if obsm in adata.obsm.keys() and varm in adata.varm.keys():
        X = DWM.T.dot(adata.varm[varm])
        X = X.dot(adata.obsm[obsm].T).T
    else:
        X = adata.X.dot(DWM)
    del DWM
    flag = np.ravel(X.sum(0)) > 0
    print("Creating AnnData object")
    gdata = anndata.AnnData(X[:, flag], obs=adata.obs, obsp=adata.obsp, obsm=adata.obsm,
                            uns={k: adata.uns[k] for k in ["neighbors"] if k in adata.uns.keys()},
                            var=gtf.loc[flag, :], dtype=np.float32)
    del X
    gdata.var.rename({"gene_id": "gene_ids"}, inplace=True, axis=1)
    gdata.var = gdata.var.loc[:, [
        "gene_ids", "feature_types", "seqname", "tss", "tts", "strand"]]
    sc.pp.normalize_total(gdata, target_sum=target_sum)
    return gdata


def annotate(adata, tsv_list):
    import numpy as np
    import pandas as pd
    L = []
    for tsv in tsv_list:
        L.append(pd.read_csv(tsv, sep="\t", index_col=0))
    df = pd.concat(L)
    df = df.loc[df.index.isin(adata.obs.index.values), :]
    for col in list(df.columns):
        if np.all(df[col].isna()):
            del df[col]
    del L
    for col in np.setdiff1d(df.columns, adata.obs.columns):
        if df[col].dtype.kind == "O":
            adata.obs[col] = ""
        else:
            adata.obs[col] = np.nan
        adata.obs.loc[df.index.values, col] = df[col].values
    return adata
