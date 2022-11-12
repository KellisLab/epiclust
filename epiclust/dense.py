from .linking import linking
from .gene_distance import peak_names_to_var

def dense(adata, output, batch_size=1000, transform=lambda x: x.loc[:, ["from", "to", "cor"]], sep="\t", float_format="%.4f", **kwargs):
    from itertools import product
    from tqdm.auto import tqdm
    import numpy as np
    import pandas as pd
    import gzip
    VN = adata.var.index.values
    var = peak_names_to_var(VN).sort_values("seqname")
    if var.shape[0] == len(VN):
        _, chrom_begin = np.unique(var["seqname"], return_index=True)
        chrom_begin = np.sort(chrom_begin)
        batches = None
        for i, begin in enumerate(chrom_begin):
            if i + 1 < len(chrom_begin):
                end = chrom_begin[i + 1]
            else:
                end = len(VN)
            df = pd.DataFrame({"begin": np.arange(begin, end, batch_size)})
            df["end"] = (df["begin"] + batch_size).clip(begin, end)
            if batches is None:
                batches = df
            else:
                batches = pd.concat((batches, df))
    else:
        batches = pd.DataFrame({"begin": np.arange(0, len(VN), batch_size)})
        batches["end"] = (batches["begin"] + batch_size).clip(0, len(VN))
    with gzip.open(output, "w") as writer:
        for i in tqdm(np.arange(batches.shape[0]), desc="row", position=0, leave=False):
            rbegin, rend = batches["begin"].values[i], batches["end"].values[i]
            for j in tqdm(np.arange(batches.shape[0]), desc="col", position=1, leave=False):
                lbegin, lend = batches["begin"].values[j], batches["end"].values[j]
                idx = pd.DataFrame(product(range(rbegin, rend), range(lbegin, lend))).values
                odf = linking(adata, VN[idx[:, 0]], VN[idx[:, 1]], **kwargs)
                transform(odf).to_csv(writer, sep="\t", index=False, header=False, float_format=float_format)
                del odf
