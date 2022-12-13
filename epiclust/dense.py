from .linking import linking
from .gene_distance import peak_names_to_var

def dense(adata, output, batch_size=1e6, transform=lambda x: x.loc[:, ["from", "to", "cor"]], sep="\t", float_format="%.4f", **kwargs):
    from itertools import product
    from tqdm.auto import tqdm
    import numpy as np
    import pandas as pd
    import gzip
    VN = adata.var.index.values
    var = peak_names_to_var(VN).sort_values(["seqname", "start"])
    if var.shape[0] == len(VN):### only accessibility; must sort
        _, chrom_begin = np.unique(var["seqname"], return_index=True)
        chrom_begin = np.sort(chrom_begin)
        batches = None
        for i, begin in enumerate(chrom_begin):
            if i + 1 < len(chrom_begin):
                end = chrom_begin[i + 1]
            else:
                end = len(VN)
            df = pd.DataFrame({"begin": [begin], "end": [end]})
            if batches is None:
                batches = df
            else:
                batches = pd.concat((batches, df))
        VN = var.index.values
    else:
        batches = pd.DataFrame({"begin": [0], "end": [len(VN)]})
    with gzip.open(output, "w") as writer:
        for i in tqdm(np.arange(batches.shape[0]), desc="row", position=0, leave=False):
            rbegin, rend = batches["begin"].values[i], batches["end"].values[i]
            for j in tqdm(np.arange(batches.shape[0]), desc="col", position=1, leave=False):
                lbegin, lend = batches["begin"].values[j], batches["end"].values[j]
                idx = pd.DataFrame(product(range(rbegin, rend), range(lbegin, lend)), columns=["right", "left"])
                idx = idx.sort_values(["right", "left"]).values
                for ibegin in tqdm(np.arange(0, idx.shape[0], batch_size), desc="chrom", position=2, leave=False):
                    iend = min(ibegin + batch_size, idx.shape[0])
                    odf = linking(adata, VN[idx[ibegin:iend, 0]], VN[idx[ibegin:iend, 1]], **kwargs)
                    transform(odf).to_csv(writer, sep="\t", index=False, header=False, float_format=float_format)
                    del odf
