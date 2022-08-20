#!/usr/bin/env python3

def load_gtf(gtf, max_score=5, min_score=1):
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

def annotate_ranges(rng, gtf_file, col="name", promoter_upstream=2000, promoter_downstream=100):
    import gtfparse
    import numpy as np
    from .gene_distance import df_to_pyranges
    gtf = gtfparse.read_gtf(gtf_file)
    exon_rng = df_to_pyranges(gtf.loc[gtf["feature"] == "exon",:], chrom="seqname", left="start", right="end")
    exonic = rng.nearest(exon_rng).df
    exonic = exonic.loc[exonic["Distance"] == 0, col].values
    del exon_rng
    ### now for gene level
    gtf = gtf.loc[gtf["feature"] == "gene",:]
    gtf["prom_left"] = gtf["start"] - promoter_upstream
    gtf.loc[gtf["strand"] == "-", "prom_left"] = gtf["end"] - promoter_downstream
    gtf["prom_right"] = gtf["start"] + promoter_downstream
    gtf.loc[gtf["strand"] == "-", "prom_right"] = gtf["end"] + promoter_upstream
    gene_rng = df_to_pyranges(gtf, chrom="seqname", left="start", right="end")
    genic = rng.nearest(gene_rng).df
    genic = genic.loc[genic["Distance"] == 0, col].values
    prom_rng = df_to_pyranges(gtf, chrom="seqname", left="prom_left", right="prom_right")
    promoter = rng.nearest(prom_rng).df
    promoter = promoter.loc[promoter["Distance"] == 0, col].values
    feature = np.asarray(np.repeat("Distal", len(rng)), dtype="<U8")
    feature[np.isin(rng.df[col].values, genic)] = "Intron"
    feature[np.isin(rng.df[col].values, exonic) & np.isin(rng.df[col].values, genic)] = "Exon"
    feature[np.isin(rng.df[col].values, promoter)] = "Promoter"
    return feature
