
def load_gtf(gtf, max_score=5, min_score=1, peaks=None):
    import pandas as pd
    import gtfparse
    import anndata
    import numpy as np
    if type(gtf) != pd.DataFrame:
        gtf = gtfparse.read_gtf(gtf)
    gtf = gtf.loc[gtf["feature"] == "gene", :].drop_duplicates(["gene_id"])
    gtf.index = gtf["gene_name"].values
    gtf.index = anndata.utils.make_index_unique(gtf.index)
    gtf["tss"] = gtf["start"].values
    gtf["tts"] = gtf["end"].values
    gtf.loc[gtf["strand"] == "-", "tss"] = gtf["end"]
    gtf.loc[gtf["strand"] == "-", "tts"] = gtf["start"]
    gtf["feature_types"] = "Gene Expression"
    diff = 1/np.abs(gtf["end"].values - gtf["start"].values).clip(1, np.inf)
    raw_score = (diff - np.min(diff))/(np.max(diff) - np.min(diff))
    gtf["gene_length_score"] = (max_score - min_score) * raw_score + min_score
    return gtf

def df_to_pyranges(var, chrom="chrom", pos="loc", margin=0, name="name", left=None, right=None):
    import pyranges
    if left is not None and right is not None:
        return pyranges.from_dict({"Chromosome": var[chrom].values,
                                       "Start": var[left].values - margin,
                                       "End": var[right].values + margin,
                                       name: var.index.values})
    else:
        return pyranges.from_dict({"Chromosome": var[chrom].values,
                                   "Start": var[pos].values - margin,
                                   "End": var[pos].values + margin,
                                   name: var.index.values})

def peak_names_to_var(peak_names, chrom="seqname", mid="loc", left="start", right="end"):
    import numpy as np
    import pandas as pd
    peak_names = np.asarray([x for x in peak_names if ":" in x and "-" in x])
    chrom_vals = np.asarray([x.split(":")[0] for x in peak_names])
    left_vals = np.asarray([int(x.split(":")[1].split("-")[0]) for x in peak_names])
    right_vals = np.asarray([int(x.split(":")[1].split("-")[1]) for x in peak_names])
    df = pd.DataFrame({chrom: chrom_vals,
                       mid: (left_vals + right_vals) / 2.,
                       left: left_vals,
                       right: right_vals,
                       "feature_types": "Peaks"})
    df.index = peak_names
    return df

def distance_weight_all(enh, gene, max_distance=1000*1000, chrom="seqname", gene_loc="tss"):
    import numpy as np
    import pandas as pd
    er = df_to_pyranges(enh, name="peak", pos="loc", chrom=chrom)
    xr = df_to_pyranges(gene, name="gene", pos=gene_loc, margin=max_distance, chrom=chrom)
    xdf = xr.join(er).df.loc[:, ["gene", "peak"]]
    sign = 2 * (gene.loc[xdf["gene"].values, "strand"].values == "+") - 1
    xdf["distance"] = enh.loc[xdf["peak"].values, "loc"].values - gene.loc[xdf["gene"].values, gene_loc].values
    xdf["distance"] *= sign
    return xdf

def index_of(values, idx):
    import numpy as np
    sort_idx = np.argsort(idx)
    values_idx = sort_idx[np.searchsorted(idx, values, sorter=sort_idx)]
    #assert np.all(idx[values_idx] == values)
    return values_idx


def genomic_distance_weight(adata, gtf, key="scm", distance=100000):
    import scipy.sparse
    import numpy as np
    if type(gtf) == str:
        gtf = load_gtf(gtf)
    gtf = gtf.loc[gtf.index.isin(adata.var.index.values), :]
    pr = df_to_pyranges(peak_names_to_var(adata.var.index.values), name="peak", pos="loc", chrom="seqname")
    dpr = df_to_pyranges(peak_names_to_var(adata.var.index.values), name="dpeak", pos="loc", chrom="seqname", margin=distance)
    gr = df_to_pyranges(gtf, name="gene", left="start", right="end", chrom="seqname")
    dgr = df_to_pyranges(gtf, name="dgene", left="start", right="end", margin=distance, chrom="seqname")
    ### 1. peak to peak
    pp = pr.join(dpr).df
    ### 2. peak to gene
    pg = pr.join(dgr).df
    ### 3. gene to gene
    gg = gr.join(dgr).df
    L = scipy.sparse.csr_matrix((np.repeat(True, pp.shape[0]),
                                  (index_of(pp["peak"].values, adata.var.index.values),
                                   index_of(pp["dpeak"].values, adata.var.index.values))),
                                dtype=bool,
                                shape=(adata.shape[1], adata.shape[1]))
    if pg.shape[0] > 0:
        L += scipy.sparse.csr_matrix((np.repeat(True, pg.shape[0]),
                                  (index_of(pg["peak"].values, adata.var.index.values),
                                   index_of(pg["dgene"].values, adata.var.index.values))),
                                 dtype=bool, shape=L.shape)
    if gg.shape[0] > 0:
        L += scipy.sparse.csr_matrix((np.repeat(True, gg.shape[0]),
                                  (index_of(gg["gene"].values, adata.var.index.values),
                                   index_of(gg["dgene"].values, adata.var.index.values))),
                                 dtype=bool, shape=L.shape)
    key_added = "%s_filter" % key
    adata.uns["filter"] = {"connectivities_key": key_added}
    adata.uns[key]["filter"] = key_added
    adata.varp[key_added] = L + L.T
