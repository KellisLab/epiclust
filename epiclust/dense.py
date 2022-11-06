from .linking import linking

def dense(adata, output, batch_size=1000, transform=lambda x: x.loc[:, ["from", "to", "cor"]], sep="\t", float_format="%.4f", **kwargs):
    from itertools import product
    from tqdm.auto import tqdm
    import numpy as np
    import pandas as pd
    import gzip
    VN = adata.var.index.values
    with gzip.open(output, "w") as writer:
        for rbegin in tqdm(np.arange(0, len(VN), batch_size), desc="row", position=0, leave=False):
            rend = min(rbegin + batch_size, len(VN))
            for lbegin in tqdm(np.arange(0, len(VN), batch_size), desc="col", position=1, leave=False):
                lend = min(lbegin + batch_size, len(VN))
                idx = pd.DataFrame(product(range(rbegin, rend), range(lbegin, lend))).values
                odf = linking(adata, VN[idx[:, 0]], VN[idx[:, 1]], **kwargs)
                transform(odf).to_csv(writer, sep="\t", index=False, header=False, float_format=float_format)
                del odf
