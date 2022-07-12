#!/usr/bin/env python3
import numpy as np
import scipy.sparse
import scipy.interpolate
from .perbin import create_bins_quantile, calc_perbin_stats

def fit_splines(adata, n_bins=50, key="scm", z=2, margin_of_error=0.05, n_bins_sample=2):
        if key not in adata.uns or "rep" not in adata.uns[key]:
                print("Run extract_module first")
                return -1
        X_adj = adata.varm[adata.uns[key]["rep"]]
        margin = X_adj[:, 0]
        X_adj = X_adj[:, 1:]
        bin_assign, edges = create_bins_quantile(margin, nbins=n_bins)
        cps = calc_perbin_stats(X_adj, bin_assign=bin_assign, z=z,
                                margin_of_error=margin_of_error,
                                n_bins_sample=n_bins_sample)
        cps["mids"] = (edges[1:] + edges[:-1]) / 2.
        adata.uns[key]["spline_info"] = cps

def build_spline(adata, key="scm", spline="mean", k=2, split=None):
        """split can be for example feature_type for linking"""
        cps = adata.uns[key]["spline_info"]
        bin_mids = cps["mids"]
        m_data = scipy.sparse.coo_matrix(cps[spline])
        counts = np.ravel(cps["counts"][m_data.row, m_data.col])
        return scipy.interpolate.SmoothBivariateSpline(x=bin_mids[m_data.row],
                                                       y=bin_mids[m_data.col],
                                                       z=m_data.data,
                                                       kx=k, ky=k,
                                                       w=np.log10(2+counts))
