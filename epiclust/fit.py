#!/usr/bin/env python3
import numpy as np
import scipy.sparse
import scipy.interpolate
from itertools import combinations_with_replacement
from .perbin import create_bins_quantile, calc_perbin_stats
from .pcor import adjust_covariates, extract_pcor_info
from .extraction import extract_rep

def _fit_bins(X_adj, margin, nbins, where_x=None, where_y=None, **kwargs):
        """where_x and where_y are for avoiding inter-indexing errors"""
        if where_x is not None:
                where_x = np.ravel(where_x)
                x_assign, x_edges = create_bins_quantile(margin[where_x], nbins=nbins)
        else:
                x_assign, x_edges = create_bins_quantile(margin, nbins=nbins)
        if where_y is not None:
                where_y = np.ravel(where_y)
                y_assign, y_edges = create_bins_quantile(margin[where_y], nbins=nbins)
        else:
                y_assign, y_edges = create_bins_quantile(margin, nbins=nbins)
        cps = calc_perbin_stats(X_adj,
                                bin_assign_row=x_assign, bin_assign_col=y_assign,
                                where_row=where_x, where_col=where_y, **kwargs)
        cps["mids_x"] = (x_edges[1:] + x_edges[:-1]) / 2.
        cps["mids_y"] = (y_edges[1:] + y_edges[:-1]) / 2.
        return cps

def fit(adata, power=0, batch=None, covariates=None,
        margin="log1p_total_counts", n_bins=50, key="epiclust",
        n_pcs=None, zero_center=True, squared_correlation=False,
        z=2, margin_of_error=0.05, n_bins_sample=1, blur=1):
        assert 0 == extract_rep(adata, power=power, margin=margin,
                                key_added=key, n_pcs=n_pcs, zero_center=zero_center)
        adata.uns[key]["squared_correlation"] = squared_correlation
        X_adj = adata.varm[adata.uns[key]["rep"]]
        margin = X_adj[:, 0]
        X_adj = X_adj[:, 1:]
        adjust_covariates(adata, covariates, key=key)
        if batch is not None and batch in adata.var.columns:
                ub, binv = np.unique(adata.var[batch].values, return_inverse=True)
                tbl = {}
                for i, j in combinations_with_replacement(np.arange(len(ub)), 2):
                        ### TODO filter i==, j== such that only fitting on filtered data
                        ib = np.ravel(np.where(i == binv))
                        jb = np.ravel(np.where(j == binv))
                        ### Use sqrt to ensure enough bins
                        nbins = np.min([n_bins,
                                        int(np.sqrt(len(ib))),
                                        int(np.sqrt(len(jb)))])
                        if nbins < 10:
                                print("Warning: Batches", ub[i], "and", ub[j], "have only", nbins, "bins")
                        cps = _fit_bins(X_adj=X_adj,
                                        margin=margin,
                                        nbins=nbins,
                                        where_x=ib,
                                        where_y=jb,
                                        z=z,
                                        margin_of_error=margin_of_error,
                                        n_bins_sample=n_bins_sample,
                                        blur=blur,
                                        squared_correlation=squared_correlation,
                                        **extract_pcor_info(adata, key=key))
                        tbl["%s %s" % (ub[i], ub[j])] = cps
                adata.uns[key]["bin_info"] = tbl
                adata.uns[key]["batches"] = ub
                adata.uns[key]["batch_key"] = batch
        else:
                adata.uns[key]["bin_info"] = _fit_bins(X_adj=X_adj,
                                                       margin=margin,
                                                       nbins=n_bins,
                                                       z=z,
                                                       margin_of_error=margin_of_error,
                                                       n_bins_sample=n_bins_sample,
                                                       blur=blur,
                                                       squared_correlation=squared_correlation,
                                                       **extract_pcor_info(adata, key=key))

def build_spline(adata, key="epiclust", spline="mean", k=2, split=None):
        """split can be for example feature_type for linking"""
        cps = adata.uns[key]["bin_info"]
        bin_mids = cps["mids"]
        m_data = scipy.sparse.coo_matrix(cps[spline])
        counts = np.ravel(cps["counts"][m_data.row, m_data.col])
        return scipy.interpolate.SmoothBivariateSpline(x=bin_mids[m_data.row],
                                                       y=bin_mids[m_data.col],
                                                       z=m_data.data,
                                                       kx=k, ky=k,
                                                       w=np.log10(2+counts))

def spline_grid_ordered(spl, x, y, **kwargs):
        Ix = np.argsort(x)
        I1x = np.argsort(Ix)
        Iy = np.argsort(y)
        I1y = np.argsort(Iy)
        data = spl(x[Ix], y[Iy], grid=True, **kwargs)
        data = data[I1x, :]
        data = data[:, I1y]
        return data
