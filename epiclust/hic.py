
def juicer_hic(df):
    import scipy.stats
    from epiclust.gene_estimation import peak_names_to_var
    var = peak_names_to_var(df["from"].values)
    df["str0"] = 0
    df["chrom0"] = var["seqname"].values
    df["pos0"] = (var["start"].values + var["end"].values) // 2
    df["frag0"] = 0
    var = peak_names_to_var(df["to"].values)
    df["str1"] = 0
    df["chrom1"] = var["seqname"].values
    df["pos1"] = (var["start"].values + var["end"].values) // 2
    df["frag1"] = 1
    df["cor"] = -scipy.stats.norm.logsf(df["cor"])
    return df.loc[:, ["str0", "chrom0", "pos0", "frag0", "str1", "chrom1", "pos1", "frag1", "cor"]]
