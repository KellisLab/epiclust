
from .pcor import extract_pcor_info
from .distance import correlation

def linking(adata, var_from_names, var_to_names, key="epiclust", min_std=0.001):
    import numpy as np
    import pandas as pd
    si = adata.uns[key]["bin_info"]
    params = {"min_std": min_std, **extract_pcor_info(adata, key=key)}
    df = pd.DataFrame({"from": np.where(adata.var.index.isin(var_from_names))[0],
                       "to": np.where(adata.var.index.isin(var_to_names))[0]})
    out = np.zeros(df.shape[0])
    X_adj = adata.varm[adata.uns[key]["rep"]]
    if "batch_key" in adata.uns[key].keys():
        df["from_batch"] = adata.var.loc[var_from_names, adata.uns[key]["batch_key"]].values
        df["to_batch"] = adata.var.loc[var_to_names, adata.uns[key]["batch_key"]].values
        ub, binv = np.unique(df.groupby(["from_batch", "to_batch"]).ngroup(), return_inverse=True)
        for i, b in enumerate(ub):
            from_batch = df["from_batch"].values[i == binv][0]
            to_batch = df["to_batch"].values[i == binv][0]
            from_batch_idx = list(adata.uns[key]["batches"]).index(from_batch)
            to_batch_idx = list(adata.uns[key]["batches"]).index(to_batch)
            if from_batch_idx <= to_batch_idx:
                key = "%s %s" % (from_batch, to_batch)
                params["mids_x"] = si[key]["mids_x"]
                params["mids_y"] = si[key]["mids_y"]
                params["mean_grid"] = si[key]["mean"]
                params["std_grid"] = si[key]["std"]
            else:
                key = "%s %s" % (to_batch, from_batch)
                params["mids_x"] = si[key]["mids_y"]
                params["mids_y"] = si[key]["mids_x"]
                params["mean_grid"] = si[key]["mean"].T
                params["std_grid"] = si[key]["std"].T
            out[i == binv] = correlation(X_adj,
                                         I_row=df["from"].values[i == binv],
                                         I_col=df["to"].values[i == binv],
                                         **params)
    else:
        params["mids_x"] = si["mids_x"]
        params["mids_y"] = si["mids_y"]
        params["mean_grid"] = si["mean"]
        params["std_grid"] = si["std"]
        out = correlation(X_adj, I_row=df["from"].values, I_col=df["to"].values, **params)
    return out
