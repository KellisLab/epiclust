import numpy as np
from tqdm.auto import tqdm
from functools import partial
import multiprocessing
from .utils import cov_to_cor_z_along

def fill_matrix(margin, X_adj, bin_assign, spline_table, z, writer):
        """bin assign could be any assignment, since spline_table takes in margin itself.
        so, bin_assign could be e.g. chromosome positioning"""
        out = []
        uniq = np.unique(bin_assign)
        nbin = len(uniq)
        order = []
        for i in range(nbin):
                for j in range(i, nbin):
                        order.append((i,j))
        for i, j in tqdm(order):
                row_indices = np.where(uniq[i] == bin_assign)[0]
                col_indices = np.where(uniq[j] == bin_assign)[0]
                cor_mat = cov_to_cor_z_along(X_adj, row_indices=row_indices, col_indices=col_indices)
                s_tables = {sname: spl(np.median(margin[row_indices]),
                                       np.median(margin[col_indices])) for sname, spl in spline_table.items()}
                cor_mat = cor_mat - s_tables["mean"]
                cor_mat = cor_mat / s_tables["std"]
                del s_tables
                cor_mat[np.equal.outer(row_indices, col_indices)] = -np.inf
                grow, gcol = np.where(cor_mat >= z)
                if len(grow) > 0 and len(gcol) == len(grow):
                        writer.add(row=row_indices[grow],
                                   col=col_indices[gcol],
                                   data=cor_mat[grow,gcol])
        return 0


def fill_matrix_bin(idx_list, margin, X_adj, bin_assign, spline_table, z, queue, batch_size=10000, correct=None):
        uniq = np.unique(bin_assign)
        i, j = idx_list
        row_indices = np.where(uniq[i] == bin_assign)[0]
        col_indices = np.where(uniq[j] == bin_assign)[0]
        cor_mat = cov_to_cor_z_along(X_adj,
                                     row_indices=row_indices,
                                     col_indices=col_indices)
        s_tables = {sname: spl(margin[row_indices], margin[col_indices])
                    for sname, spl in spline_table.items()}
        cor_mat = cor_mat - s_tables["mean"]
        cor_mat = cor_mat / s_tables["std"]
        if correct is not None:
                cor_mat = correct(cor_mat, row_indices, col_indices)
        cor_mat[np.equal.outer(row_indices, col_indices)] = -np.inf
        grow, gcol = np.where(cor_mat >= z)
        for begin in range(0, len(grow), batch_size):
                end = min(begin+batch_size, len(grow))
                queue.put({"row": row_indices[grow[begin:end]],
                           "col": col_indices[gcol[begin:end]],
                           "data": cor_mat[grow[begin:end], gcol[begin:end]]})
        queue.put({"done": True})
        return 0

def write_from_queue(writer, queue, n_items):
        n_done = 0
        t = tqdm(total=n_items)
        while n_done < n_items:
                item = queue.get()
                if "done" in item.keys():
                        n_done += 1
                        t.update(1)
                else:
                        writer.add(**item)
        t.close()
        return 0

def fill_matrix_parallel(margin, X_adj, bin_assign, spline_table, z, writer, correct=None, batch_size=1000):
        """bin assign could be any assignment, since spline_table takes in margin itself.
        so, bin_assign could be e.g. chromosome positioning"""
        uniq = np.unique(bin_assign)
        order = []
        for i in range(len(uniq)):
                for j in range(i, len(uniq)):
                        order.append((i, j))
        with multiprocessing.Manager() as manager:
                queue = manager.Queue()
                wproc = multiprocessing.Process(target=write_from_queue,
                                                args=(writer, queue, len(order)))
                wproc.start()
                func = partial(fill_matrix_bin, margin=margin, X_adj=X_adj,
                               bin_assign=bin_assign, spline_table=spline_table,
                               z=z, queue=queue, batch_size=batch_size)
                with multiprocessing.Pool() as pool:
                        out = pool.map(func, order)
                wproc.join()
        return 0
