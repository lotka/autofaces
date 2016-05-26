import collections
import os
import numpy as np
import h5py

class FileHDF5(object):
    def __init__(self, path_to_file, name_item):
        self.__path_to_file = path_to_file
        self.__name_item = name_item
        self.print_enabled = True
        self.__shape = None

    def __getitem__(self, item):
        self._print('read file ' + self.__path_to_file)
        f = h5py.File(self.__path_to_file, 'r')
        hdf5_item = f.get(self.__name_item)
        # update shape; should have low cost
        self.__shape = hdf5_item.shape
        hdf5_item = hdf5_item[item]
        data = np.array(hdf5_item)
        f.close()
        return data

    def _print(self, str_):
        if self.print_enabled:
            classname = type(self).__name__
            print("[" + classname + "] " + str_)

    def __len__(self):
        self.__load_info()
        return self.__shape[0]

    def __repr__(self):
        classname = type(self).__name__
        string = "{}(path_to_file={}, name_item={})".format(classname, repr(self.__path_to_file),
                                                            repr(self.__name_item))
        return string

    def __load_info(self):
        if self.__shape is None:
            f = h5py.File(self.__path_to_file, 'r')
            hdf5_item = f.get(self.__name_item)
            self.__shape = hdf5_item.shape
            f.close()

    @property
    def shape(self):
        self.__load_info()
        return self.__shape

    @property
    def ndims(self):
        self.__load_info()
        return len(self.__shape)


class TransposeFront(object):
    def __init__(self, obj, ndims, front, reverse=False):
        assert front < ndims
        self.obj = obj
        self.ndims = ndims
        self.front = front
        self.reverse = reverse

    def __order(self):
        order = list(range(self.ndims))
        front_elem = order.pop(self.front)
        if self.reverse:
            order.reverse()
        order.insert(0, front_elem)
        return order

    def __order_without(self, keep_dim):
        order = []
        next = 0
        for k in keep_dim:
            if k:
                order.append(next)
                next += 1
            else:
                order.append(None)
        front_elem = order.pop(self.front)
        if self.reverse:
            order.reverse()
        order.insert(0, front_elem)
        order = [o for o in order if o is not None]
        return order

    def __order_inv(self):
        order = self.__order()
        order_inv = [0] * len(order)
        for i in range(len(order)):
            order_inv[order[i]] = i
        return order_inv

    @property
    def shape(self):
        shape_inner = self.obj.shape
        # lazy checking of consistency
        assert self.ndims == len(shape_inner)
        shape = tuple(shape_inner[i] for i in self.__order())
        return shape

    def __getitem__(self, item):
        if not isinstance(item, collections.Iterable):
            item = (item,)
        assert len(item) <= self.ndims
        item_all = [np.s_[:],] * self.ndims
        keep_dim = np.ones(self.ndims, dtype=bool)
        for n, it in enumerate(item):
            item_all[n] = it
            if isinstance(it, int):
                keep_dim[n] = False
        item_inner = tuple(item_all[i] for i in self.__order_inv())
        data = self.obj[item_inner]
        data = np.transpose(data, axes=self.__order_without(keep_dim))
        return data

    def __repr__(self):
        classname = type(self).__name__
        string = "{}(obj={}, ndims={}, front={}, reverse={})".format(classname, repr(self.obj), repr(self.ndims),
                                                                     repr(self.front), repr(self.reverse))
        return string

    def __len__(self):
        return self.shape[0]


class FileArray(object):
    def __init__(self):
        # FileHDF5 arguments
        self.parts_path_to_file = None
        self.name_item = None
        # TransposeFront arguments
        self.ndims = None
        self.front = None
        self.reverse = None
        # other arguments
        self.K = None # should have same shape as single data sample

    def get_path_to_file(self, *args):
        path_to_file_raw = os.path.join(*(self.parts_path_to_file))
        path_to_file = path_to_file_raw.format(*args)
        return path_to_file

    def get_array_id(self, *args):
        path_to_file = self.get_path_to_file(*args)
        file_hdf5 = FileHDF5(path_to_file, self.name_item)
        file_hdf5_transpose= TransposeFront(file_hdf5, self.ndims, self.front, self.reverse)
        return file_hdf5_transpose

    def __getitem__(self, item):
        return self.get_array_id(item)


class IndicesCollection(object):
    def __init__(self, coll=None, id_array=None):
        self.coll = coll
        if id_array is not None:
            self.from_id_array(id_array)

    def __repr__(self):
        return repr(self.coll)

    def __str__(self):
        return str(self.coll)

    def getitem(self, obj):
        result = []
        ids = []
        for c in self.coll:
            obj_inner = obj
            if not isinstance(c, collections.Iterable):
                c = (c,)
            ids_act = []
            for ind in c:
                if isinstance(ind, slice):
                    s = ind.indices(len(obj_inner))
                    ids_act.append((np.arange(s[0], s[1], s[2]),))
                else:
                    ids_act.append(ind)

                obj_inner = obj_inner[ind]
            result.append(obj_inner)
            ids.append(ids_act)
        result = np.concatenate(result)
        id_array = IndicesCollection(ids).to_id_array()
        return result, id_array

    def to_id_array(self):
        id_all = np.zeros((0,0), dtype=int)
        for c in self.coll:
            id_prefix = np.zeros((1, 0), dtype=int)
            if not isinstance(c, collections.Iterable):
                c = (c,)
            for ind in c:
                ind = np.array(ind, dtype=int)
                ind = ind.reshape(-1,1)
                if ind.shape[0] > 1:
                    id_prefix = np.broadcast_to(id_prefix, (ind.shape[0], id_prefix.shape[1]))
                if id_prefix.shape[0] > 1:
                    ind = np.broadcast_to(ind, (id_prefix.shape[0], ind.shape[1]))
                id_prefix = np.concatenate([id_prefix, ind], axis=1)
            if id_all.shape[0] == 0:
                id_all = id_prefix
            else:
                id_all = np.concatenate([id_all, id_prefix], axis=0)
        return id_all

    def from_id_array(self, id_all):
        self.coll = []
        self.__from_id_array_rec(id_all)

    def __from_id_array_rec(self, id_all):
        if id_all.shape[0] == 0:
            # finished
            return

        if id_all.shape[1] == 1:
            # all within single indices
            self.coll.append([(id_all,)])
            return

        # beginning of first row is the next id
        id_next = id_all[0,:-1]
        ind = np.all(id_all[:,:-1] == id_next, axis=1)
        ind_tuple = [x for x in id_next]
        ind_tuple.append((id_all[ind,-1],))
        self.coll.append(ind_tuple)
        id_all = id_all[~ind, :]
        self.__from_id_array_rec(id_all)