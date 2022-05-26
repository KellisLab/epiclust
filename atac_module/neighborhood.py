
import pyranges
import pandas as pd
import numpy as np
def aggregate_chrom(var, distance=100000, offset=0):
    maxchrom = var.groupby("chrom").aggregate(maxloc=("loc", "max")).reset_index()
    L = []
    for chrom, maxloc in zip(maxchrom["chrom"].values, maxchrom["maxloc"].values):
        begin = np.arange(offset-distance, maxloc+distance, distance).clip(0, np.inf).astype(int)
        base = pd.DataFrame({"Chromosome": chrom,
                             "Start": begin,
                             "End": begin + distance})
        L.append(base)
    df = pd.concat(L).drop_duplicates()
    df["neighborhood"] = np.arange(df.shape[0])
    return pyranges.from_dict({"Chromosome": df["Chromosome"].values,
                               "Start": df["Start"].values,
                               "End": df["End"].values,
                               "neighborhood": df["neighborhood"].values})

def find_neighborhoods(var, distance=100000, offset=0):
    rgr = aggregate_chrom(var, distance=distance, offset=offset)
    vgr = pyranges.from_dict({"Chromosome": var["chrom"].values,
                              "Start": var["loc"].values,
                              "End": var["loc"].values,
                              "name": var.index.values})
    fn = rgr.join(vgr).df[["name", "neighborhood"]].drop_duplicates("name")
    nidx = pd.DataFrame(np.ones(var.shape[0], dtype=int) * -1,
                        index=var.index.values, columns=["n_idx"])
    nidx.loc[fn["name"].values, "n_idx"] = fn["neighborhood"].values
    return nidx["n_idx"].values
