#!/usr/bin/env python3
import numpy as np
import scipy.sparse
import scipy.interpolate


class Spline:
        def __init__(self, bin_assign, bin_edges, m_data, m_count, k=2, a_min=-np.inf, a_max=np.inf):
                bin_mids = (bin_edges[1:] + bin_edges[:-1])/2
                m_data = scipy.sparse.coo_matrix(m_data)
                self.total_mean = np.mean(m_data.data)
                count_mean = np.ravel(m_count[m_data.row, m_data.col])
                self.spl = scipy.interpolate.SmoothBivariateSpline(x=bin_mids[m_data.row],
                                                                   y=bin_mids[m_data.col],
                                                                   z=m_data.data - self.total_mean,
                                                                   kx=k, ky=k,
                                                                   w=np.log10(2+count_mean))
                self.a_min = a_min
                self.a_max = a_max

        def __call__(self, x, y):
                Ix = np.argsort(x)
                I1x = np.argsort(Ix)
                Iy = np.argsort(y)
                I1y = np.argsort(Iy)
                data = self.spl(x[Ix], y[Iy])
                data = data[I1x, :]
                data = data[:, I1y]
                return np.clip(data + self.total_mean, a_min=self.a_min, a_max=self.a_max)
