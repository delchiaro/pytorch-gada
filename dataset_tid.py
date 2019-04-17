import time
from copy import copy
from pathlib import Path
import pandas as pd
import numpy as np
from imageio import imread

import multiprocessing as mp

from gada.gadaset import GADAsetFactory


class ImageLoader:
    ref_images = {}
    dist_images = {}

    def __init__(self, dist_path, ref_path):
        self.dist_path = dist_path
        self.ref_path = ref_path

    def __call__(self, t):
        index, dist_fname, ref_fname = t
        dist = np.array(imread(self.dist_path / dist_fname)).swapaxes(2,1).swapaxes(1,0)
        ref = np.array(imread(self.ref_path / ref_fname)).swapaxes(2,1).swapaxes(1,0)
        return index, dist, ref



class TID2013(GADAsetFactory):

    def __init__(self, dataset_path, init=False):
        super().__init__()
        self.path = Path(dataset_path)
        if init:
            self.init()

    def init(self):
        lines = open(self.path / 'mos_with_names.txt', 'r').read().split('\n')
        lines.remove('')
        dist_im_names = [l.split(' ')[1] for l in lines]
        ref_im_names = [n.split('_')[0].upper() + '.BMP' for n in dist_im_names]
        dist_types = [int(n.split('.')[0].split('_')[1]) for n in dist_im_names]
        dist_level = [int(n.split('.')[0].split('_')[2]) for n in dist_im_names]

        # Pandas DataFrame:
        metadata = {TID2013.COL_DISTORTED_NAME: dist_im_names,
                    TID2013.COL_REFERENCE_NAME: ref_im_names,
                    TID2013.COL_DISTORTION_TYPE: dist_types,
                    TID2013.COL_DISTORTION_LEVEL: dist_level}
        self.data = pd.DataFrame(data=metadata)
        return self

    @property
    def metadata_cols(self):
        return TID2013.COLS_METADATA

    def random_idx(self, n):
        return np.random.permutation(self.data.index)[:n]

    def _split_frame(self, idx):
        return self.data.iloc[idx] # pandas DataFrame
        #return self.data.isel(index=idx) # xarray Dataset

    def _random_split_frame(self, n):
        return self._split_frame(self.random_idx(n))

    def split_frame(self, idx=None, n_random=None):
        assert (idx is None) != (n_random is None)
        if idx is not None:
            return self._split_frame(idx)
        else:
            return self._random_split_frame(n_random)

    def trainval_split(self, train_idx=None, train_n_random=None):
        train = self.split(train_idx, train_n_random)
        val_idx = list(set(self.data.index) - set(train.data.index))
        val = self.split(val_idx)
        return train, val

    def split(self, idx=None, n_random=None, inplace=False):
        data = self.split_frame(idx, n_random)
        if inplace:
            self.data = data
            return self
        else:
            other = copy(self)
            other.data = data
            return other

    def load_images(self, inplace=True, verbose=False, threads=mp.cpu_count()):
        return super().load_images(inplace, verbose, **{'threads': threads})

    def _load_images(self, inplace=True, verbose=False, threads=mp.cpu_count()):
        if verbose:
            print(f'Loading image files from folder {self.path}  ...')
        now = time.time()
        self._image_loaded = True
        lists = [self.data.index, list(self.data[TID2013.COL_DISTORTED_NAME]), list(self.data[TID2013.COL_REFERENCE_NAME])]
        indices = self.data.index
        args = []
        for i, d, r in zip(*lists):
            if i in indices:
                args.append((i, d, r))
        dist_path = self.path/'distorted_images'
        ref_path = self.path/'reference_images'

        #import time
        #a = time.time()
        proc = ImageLoader(dist_path, ref_path)
        pool = mp.Pool(threads)
        res = pool.map(proc, args)
        #print(time.time() - a)

        idx = [r[0] for r in res]
        dist = [r[1] for r in res]
        ref = [r[2] for r in res]
        if verbose:
            print(f'DONE: images loaded in {time.time()-now} seconds.')

        if inplace:
            self.data[TID2013.COL_REFERENCE_IMAGES] = ref
            self.data[TID2013.COL_DISTORTED_IMAGES] = dist
            return self
        else:
            return np.array(ref, dtype=np.uint8), np.array(dist, dtype=np.uint8)

    def _drop_images(self):
        self._image_loaded = False
        try:
            self.data.drop(columns=[TID2013.COL_DISTORTED_IMAGES, TID2013.COL_REFERENCE_IMAGES], inplace=True)
        except:
            pass

    def print_rows(self):
        for id, row in tid.data.iterrows():
            print(f"id:         {id}")
            print(f"Reference:  {row[TID2013.COL_REFERENCE_NAME]}")
            print(f"Distorted:  {row[TID2013.COL_DISTORTED_NAME]}")
            print(f"Dist type:  {row[TID2013.COL_DISTORTION_TYPE]}")
            print(f"Dist level: {row[TID2013.COL_DISTORTION_LEVEL]}\n")

    def print_data(self):
        from tabulate import tabulate
        headers = [TID2013.COL_INDEX] + self.metadata_cols
        not_metadata = set(self.data.columns) - set(self.metadata_cols)

        ndarrays = [col for col in not_metadata if hasattr(self.data[col][list(self.data[col].keys())[0]], 'shape')]
        not_metadata = not_metadata - set(ndarrays)

        data = [[id] + [row[col] for col in self.metadata_cols] + [row[col].shape for col in ndarrays] +\
                [type(row[col]) for col in not_metadata] for id, row in self.data.iterrows()] # with pandas DataFrame
        #data = [[id] + [data_var.data for data_var in self.data.isel(index=id).data_vars.values()] for id in self.data.index] # with xarray
        return tabulate(data, headers)

    def reference_images(self):
        if TID2013.COL_REFERENCE_IMAGES in self.data.columns:
            return np.stack(self.data[TID2013.COL_REFERENCE_IMAGES])
        #return self.data[COL_REFERENCE_IMAGES]

    def distorted_images(self):
        if TID2013.COL_REFERENCE_IMAGES in self.data.columns:
            return np.stack(self.data[TID2013.COL_DISTORTED_IMAGES])
        #return self.data[COL_DISTORTED_IMAGES]

    def distortion_types(self) -> np.ndarray:
        return np.expand_dims(np.array(self.data[TID2013.COL_DISTORTION_TYPE], dtype=np.uint8), axis=-1)
        #return self.data[COL_DISTORTION_TYPE]

    def distortion_level(self) -> np.ndarray:
        return np.expand_dims(np.array(self.data[TID2013.COL_DISTORTION_LEVEL], dtype=np.uint8), axis=-1)
        #return self.data[COL_DISTORTION_LEVEL]

    def indices(self):
        return np.expand_dims(np.array(self.data.index, dtype=np.uint8), axis=-1)
        #return self.data.index


    def save(self, path):
        self.data.to_hdf(path, 'tid')
        return self

    def load(self, path):
        self.data = pd.read_pickle(path)
        return self

    def save_split_file(self, path):
        f = open(path, 'w')
        for idx in sorted(self.data.index):
            f.write(str(idx) + '\n')
        f.close()

    def load_split_file(self, path, inplace=False):
        idx = open(path, 'r').read().strip().split('\n')
        idx = [int(id) for id in idx]
        return self.split(idx, inplace=inplace)

    def apply_split_file(self, path):
        return self.load_split_file(path, inplace=True)

    def __str__(self):
        return str(f"path: {self.path}\ndata:\n{self.print_data()}")





if __name__ == "__main__":

    #%%
    tid = TID2013('/mnt/2tb/datasets/IQA/tid2013/').init()
    tid.load_images()

    print(tid)

    train, val = tid.trainval_split(train_n_random=2200)
    tid.save('tid.pickle')
    train.save('tid_train_2200.pickle')
    val.save('tid_test_800.pickle')

    #%%
    tid_train = TID2013('/mnt/2tb/datasets/IQA/tid2013/').load('tid_train_2200.pickle')
    tid_test = TID2013('/mnt/2tb/datasets/IQA/tid2013/').load('tid_test_800.pickle')


    #%%
    tid_train.save_split_file('tid_train_2200.split')
    tid_test.save_split_file('tid_test_800.split')


    #%%
    tid = TID2013('/mnt/2tb/datasets/IQA/tid2013/').load('tid.pickle')
    tid_train = tid.load_split_file('tid_train_2200.split')
    tid_test = tid.load_split_file('tid_test_800.split')