import h5py
class H5Writer:
        def __init__(self, filename, names, chunks=1000):
                self.h5 = h5py.File(filename, "w")
                self.h5["name"] = names
                ms = len(names)*len(names)
                self.h5.create_dataset("row",
                                       (0,),
                                       maxshape=(ms,),
                                       chunks=(chunks,),
                                       compression="gzip",
                                       dtype="int64")
                self.h5.create_dataset("col",
                                       (0,),
                                       maxshape=(ms,),
                                       chunks=(chunks,),
                                       compression="gzip",
                                       dtype="int64")
                self.h5.create_dataset("data",
                                       (0,),
                                       maxshape=(ms,),
                                       chunks=(chunks,),
                                       compression="gzip",
                                       dtype="float64")
                self.cur_len = 0
        def add(self, row, col, data):
                data_len = min(min(len(row), len(col)), len(data))
                old_len = self.cur_len
                new_len = old_len + data_len
                self.cur_len = new_len
                self.h5["row"].resize((self.cur_len,))
                self.h5["col"].resize((self.cur_len,))
                self.h5["data"].resize((self.cur_len,))
                self.h5["row"][old_len:new_len] = row
                self.h5["col"][old_len:new_len] = col
                self.h5["data"][old_len:new_len] = data
        def __del__(self):
                self.h5.close()
