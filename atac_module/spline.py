#!/usr/bin/env python3
import numpy as np
import scipy.sparse
import scipy.interpolate

def build_spline(bin_assign, bin_edges, m_data, m_count, k=2, a_min=-np.inf, a_max=np.inf):
        """build a closure that encapsulates a spline-function for calculating a grid of numbers"""
        bin_mids = (bin_edges[1:] + bin_edges[:-1])/2
        m_data = scipy.sparse.coo_matrix(m_data)
        total_mean = np.mean(m_data.data)
        count_mean = np.ravel(m_count[m_data.row, m_data.col])
        spl = scipy.interpolate.SmoothBivariateSpline(x=bin_mids[m_data.row],
                                                      y=bin_mids[m_data.col],
                                                      z=m_data.data - total_mean,
                                                      kx=k, ky=k, w=np.log10(2+count_mean))
        def build_spline_grid(x, y):
                """ in order to create x times y matrix, have to have
                X and Y in order, then can re-order to original coordinates"""
                Ix = np.argsort(x)
                I1x = np.argsort(Ix)
                Iy = np.argsort(y)
                I1y = np.argsort(Iy)
                data = spl(x[Ix], y[Iy])
                data = data[I1x,:]
                data = data[:,I1y]
                return np.clip(data + total_mean, a_min=a_min, a_max=a_max)
        return build_spline_grid
