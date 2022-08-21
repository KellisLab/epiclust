
def df_to_pyranges(var, chrom="chrom", pos="loc", margin=0,
                   name="name", left=None, right=None, **kwargs):
    import pyranges
    tbl = {k: var[v].values for k, v in kwargs.items()}
    tbl["Chromosome"] = var[chrom].values
    tbl[name] = var.index.values
    if left is not None and right is not None:
        tbl["Start"] = var[left].values - margin
        tbl["End"] = var[right].values + margin
    else:
        tbl["Start"] = var[pos].values - margin
        tbl["End"] = var[pos].values + margin
    return pyranges.from_dict(tbl)


def peak_names_to_var(peak_names, chrom="seqname",
                      mid="loc", left="start", right="end"):
    import numpy as np
    import pandas as pd
    peak_names = peak_names[pd.Series(
        peak_names).str.match(r"^[^:]+:[0-9]+-[0-9]+$")]
    chrom_vals = np.asarray([x.split(":")[0] for x in peak_names])
    left_vals = np.asarray([int(x.split(":")[1].split("-")[0])
                           for x in peak_names])
    right_vals = np.asarray(
        [int(x.split(":")[1].split("-")[1]) for x in peak_names])
    df = pd.DataFrame({chrom: chrom_vals,
                       mid: (left_vals + right_vals) / 2.,
                       left: left_vals,
                       right: right_vals,
                       "feature_types": "Peaks"})
    df.index = peak_names
    return df


def distance_weight_all(enh, gene, max_distance=1000 *
                        1000, chrom="seqname", gene_loc="tss"):
    """enh from peak_names_to_var, gene from load_gtf().loc[genes,:]"""
    import numpy as np
    import pandas as pd
    er = df_to_pyranges(enh, name="peak", pos="loc", chrom=chrom)
    xr = df_to_pyranges(gene, name="gene", pos=gene_loc,
                        margin=max_distance, chrom=chrom)
    xdf = xr.join(er).df.loc[:, ["gene", "peak"]]
    sign = 2 * (gene.loc[xdf["gene"].values, "strand"].values == "+") - 1
    xdf["distance"] = enh.loc[xdf["peak"].values, "loc"].values - \
        gene.loc[xdf["gene"].values, gene_loc].values
    xdf["distance"] *= sign
    return xdf


def distance_weight(enh, gene, max_distance=1000 * 1000, margin=5000,
                    pos="loc", chrom="seqname", left="start", right="end", closest_gene=True):
    import numpy as np
    import pandas as pd
    er = df_to_pyranges(enh, name="peak", pos=pos, chrom=chrom)
    xr = df_to_pyranges(gene, name="gene", left=left,
                        right=right, margin=margin, chrom=chrom)
    lr = df_to_pyranges(gene, name="gene", pos=left,
                        margin=max_distance, chrom=chrom)
    rr = df_to_pyranges(gene, name="gene", pos=right,
                        margin=max_distance, chrom=chrom)
    xdf = xr.join(er).df.loc[:, ["gene", "peak"]]
    # gene body distance is 0
    xdf["distance"] = 0
    ldf = lr.join(er).df.loc[:, ["gene", "peak"]]
    # LHS distance is difference to LHS of gene body
    ldf["distance"] = np.abs(
        enh.loc[ldf["peak"].values, pos].values - gene.loc[ldf["gene"].values, left].values)
    rdf = lr.join(er).df.loc[:, ["gene", "peak"]]
    # RHS distance is difference to RHS of gene body
    rdf["distance"] = np.abs(
        enh.loc[rdf["peak"].values, pos].values - gene.loc[rdf["gene"].values, right].values)
    # Get shortest distance
    df = pd.concat([xdf, ldf, rdf]).sort_values(
        "distance").drop_duplicates(["peak", "gene"])
    if closest_gene:
        # Don't count peaks twice unless in gene overlapping body/promoter
        df = pd.concat([df.loc[df["distance"] == 0, :],
                        df.drop_duplicates(["peak"])]).drop_duplicates(["peak", "gene"])
    df["weight"] = np.exp(-1 - df["distance"] / 5000)
    return df
