import numbers
import random
import time
from copy import copy, deepcopy
from os.path import join
import os

import torch
from abc import ABC, abstractmethod
import numpy as np
from torch.utils.data import TensorDataset, Dataset, ConcatDataset
from torch.nn import functional as NNF
import multiprocessing as mp
import pandas as pd
from typing import Union, Tuple, List


class RandomCrop(object):
    def __init__(self, size, padding=0, pad_if_needed=False, jpeg_grid_alignment=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.grid_alignment = 8 if jpeg_grid_alignment else 1

    @staticmethod
    def get_params(img_tensor, output_size, grid_alignment=1):
        w_dim=-1
        h_dim=-2
        h = img_tensor.shape[h_dim]
        w = img_tensor.shape[w_dim]

        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, (h-th)//grid_alignment)*grid_alignment
        j = random.randint(0, (w-tw)//grid_alignment)*grid_alignment

        return i, j, th, tw

    def __call__(self, img_tensor: torch.Tensor):
        w_dim = -1
        h_dim = -2
        from torchvision.transforms import functional as F
        if self.padding > 0:
            img_tensor = NNF.pad(img_tensor, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img_tensor.shape[w_dim] < self.size[h_dim]:
            img_tensor = NNF.pad(img_tensor, pad=(int((1 + self.shape[h_dim] - img_tensor.shape[w_dim]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img_tensor.size[h_dim] < self.shape[w_dim]:
            img_tensor = F.pad(img_tensor, (0, int((1 + self.shape[w_dim] - img_tensor.shape[h_dim]) / 2)))

        i, j, h, w = self.get_params(img_tensor, self.size, self.grid_alignment)
        return img_tensor[:, i:i+h, j:j+w]
        #return F.crop(img_tensor, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)



class UnisonRandomCrop(RandomCrop):
    def __init__(self, size, padding=0, pad_if_needed=False, jpeg_grid_alignment=False):
        super().__init__(size, padding, pad_if_needed, jpeg_grid_alignment)

    def __call__(self, img_tensor_tuple: Tuple[torch.Tensor]):
        channels = [img_tensor.shape[0] for img_tensor in img_tensor_tuple]
        joined_img_tensor = torch.cat(img_tensor_tuple, dim=0)
        joined_img_tensor_cropped = super().__call__(joined_img_tensor)
        img_tensors_cropped = []
        ch_st = 0
        for ch in channels:
            img_tensors_cropped.append(joined_img_tensor_cropped[ch_st:ch_st+ch])
            ch_st += ch
        return tuple(img_tensors_cropped)




class ToDevice(object):
    def __init__(self, device):
        self.device = device
    def __call__(self, tensor):
        return tensor.to(self.device)
    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToTensor(object):
    def __call__(self, pic):
        from torchvision.transforms import functional as F
        return F.to_tensor(pic)
    def __repr__(self):
        return self.__class__.__name__ + '()'

from imageio import imread

class ImageLoader:
    ref_images = {}
    dist_images = {}

    def __init__(self, ref_path, dist_path):
        self.ref_path = ref_path
        self.dist_path = dist_path

    def __call__(self, t):
        index, ref_fname, dist_fname = t
        ref = np.array(imread(self.ref_path / ref_fname))
        dist = np.array(imread(self.dist_path / dist_fname))
        return index, ref, dist


class StackDataset(Dataset):
    def __init__(self, datasets):
        assert all(len(datasets[0]) == len(d) for d in datasets)
        self.datasets = datasets

    def __getitem__(self, index):
        #return tuple(d[index] for d in self.datasets)
        return tuple(col for d in self.datasets for col in d[index])

    def __len__(self):
        return len(self.datasets[0])


class ListOfNumpyArrayDataset(Dataset):
    def __init__(self, *array_lists, transform_fn=None, global_transform_fn=None):
        assert all(len(array_lists[0]) == len(img_list) for img_list in array_lists)
        self.array_lists = array_lists
        self.transform_fn = transform_fn
        self.global_transform_fn = global_transform_fn

    def __getitem__(self, index):
        if self.transform_fn is not None:
            result = tuple(self.transform_fn(img_list[index]) for img_list in self.array_lists)
        else:
            result = tuple(img_list[index] for img_list in self.array_lists)
        if self.global_transform_fn is not None:
            result = self.global_transform_fn(result)
        return result

    def __len__(self):
        return len(self.array_lists[0])


class NumpyArrayDataset(Dataset):

    def __init__(self, *arrays, transform_fn=None, global_transform_fn=None):
        assert all(len(arrays[0]) == len(array) for array in arrays)
        self.numpy_arrays = arrays
        self.transform_fn = transform_fn
        self.global_transform_fn = global_transform_fn


    def __getitem__(self, index):
        if self.transform_fn is not None:
            result = tuple(self.transform_fn(array[index]) for array in self.numpy_arrays)
        else:
            result = tuple(array[index] for array in self.numpy_arrays)
        if self.global_transform_fn is not None:
            result = self.global_transform_fn(result)
        return result

    def __len__(self):
        return len(self.numpy_arrays[0])






class GADAsetFactory(ABC):
    COL_INDEX = 'index'
    COL_DISTORTED_NAME = 'distorted_name'
    COL_REFERENCE_NAME = 'reference_name'
    COL_DISTORTION_CATEG = 'distortion_type'
    COL_DISTORTION_LEVEL = 'distortion_level'

    COL_REFERENCE_INDEX = 'reference_index'

    #COLS_METADATA = [COL_DISTORTED_NAME, COL_REFERENCE_NAME, COL_DISTORTION_TYPE, COL_DISTORTION_LEVEL, COL_REFERENCE_INDEX]
    COL_DISTORTED_IMAGES = 'distorted_images'
    COL_REFERENCE_IMAGES = 'reference_images'

    @staticmethod
    def set_display_options(max_rows=500, max_cols=500, max_width=1000):
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        pd.set_option('mode.chained_assignment', None) # improve performance? Fix bug with pycharm debugger?

    def __init__(self, load_dataset=False):
        self.data = None
        self.ref_data = None
        self._image_loaded = False
        if load_dataset:
            self.init()

    @abstractmethod
    def _init(self, metadata, **kwargs) -> dict: pass


    def init(self, first_ref_image_id=1, **kwargs):
        metadata = {GADAsetFactory.COL_DISTORTED_NAME: None,
                    GADAsetFactory.COL_REFERENCE_NAME: None,
                    GADAsetFactory.COL_DISTORTION_CATEG: None,
                    GADAsetFactory.COL_DISTORTION_LEVEL: None}
        metadata = self._init(metadata, **kwargs)
        ref_image_names = sorted(set(metadata[GADAsetFactory.COL_REFERENCE_NAME]))
        self.ref_data = pd.DataFrame(data={GADAsetFactory.COL_REFERENCE_NAME: ref_image_names}, index=range(first_ref_image_id, len(ref_image_names)+first_ref_image_id))
        reference_name2index = {name: index for index, name in zip(self.ref_data.index, self.ref_data[GADAsetFactory.COL_REFERENCE_NAME])}
        metadata[GADAsetFactory.COL_REFERENCE_INDEX] = [reference_name2index[ref_name] for ref_name in metadata[GADAsetFactory.COL_REFERENCE_NAME]]
        self.data = pd.DataFrame(data=metadata)
        return self

    def load_images(self, inplace=True, threads=mp.cpu_count(), verbose=False, **kwargs) -> (List[np.ndarray], List[np.ndarray]):
        """:param inplace: if True the images will be loaded inside the object and this function return self,
                  otherwise this function return the loaded images.
           :param threads:
           :param verbose:
           :param kwargs:
           :return: return two list of 3d-arrays (or two 4d-arrays), each 3d-array is an image with (Y,X,CHANNEL) fromat
                    where channel should be 3 (RGB) and each value should be in 0-255.
                    The first list should contains the Reference Images, the second one the Distorted Images.
           """
        self._image_loaded=True
        start = time.time()
        ref_imgs_array, dist_imgs_array = self._load_images(threads, verbose, **kwargs)
        if verbose:
            print(f'LOAD IMAGES - DONE: images loaded in {time.time()-start} seconds.')

        if inplace:
            self.data[GADAsetFactory.COL_REFERENCE_IMAGES] = [img for img in ref_imgs_array]
            self.data[GADAsetFactory.COL_DISTORTED_IMAGES] = [img for img in dist_imgs_array]
            return self
        else:
            return ref_imgs_array, dist_imgs_array


    @abstractmethod
    def _load_images(self, threads, verbose, **kwargs) -> (List[np.ndarray], List[np.ndarray]):
        """

        :param threads:
        :param verbose:
        :param kwargs:
        :return: Should return two list of 3d-arrays (or two 4d-arrays), each 3d-array is an image with (Y,X,CHANNEL) fromat
                 where channel should be 3 (RGB) and each value should be in 0-255.
                 The first list should contains the Reference Images, the second one the Distorted Images.
        """
        pass


    def _load_images_helper(self, ref_imgs_path, distorted_imgs_path, threads, verbose) -> (np.ndarray, np.ndarray):
        ref_path = ref_imgs_path
        dist_path = distorted_imgs_path
        if verbose:
            print(f'Loading image files from folder {dist_path} and {ref_path} ...')
        lists = [self.data.index, list(self.data[GADAsetFactory.COL_REFERENCE_NAME]), list(self.data[GADAsetFactory.COL_DISTORTED_NAME])]
        indices = self.data.index
        args = []
        for i, ref, dist in zip(*lists):
            if i in indices:
                args.append((i, ref, dist))

        result = mp.Pool(threads).map(ImageLoader(ref_path, dist_path), args)

        # idx = [r[0] for r in res]
        ref = [res[1] for res in result]
        dist = [res[2] for res in result]
        try:
            ref = np.array(ref, dtype=np.uint8)
        except:
            pass
        try:
            dist = np.array(dist, dtype=np.uint8)
        except:
            pass
        return ref, dist


    def drop_images(self):
        self._image_loaded = False
        try:
            self.data.drop(columns=[GADAsetFactory.COL_DISTORTED_IMAGES, GADAsetFactory.COL_REFERENCE_IMAGES], inplace=True)
        except:
            pass


    @property
    def image_loaded(self):
        return self._image_loaded


    @property
    def metadata_cols(self):
        return GADAsetFactory.COLS_METADATA



    def random_idx(self, n):
        return np.random.permutation(self.data.index)[:n]

    def _split_frame(self, idx):
        return self.data.loc[idx]

    def _random_split_frame(self, n):
        return self._split_frame(self.random_idx(n))

    def split_frame(self, idx=None, n_random=None):
        assert (idx is None) != (n_random is None)
        if idx is not None:
            return self._split_frame(idx)
        else:
            return self._random_split_frame(n_random)

    def split(self, idx=None, n_random=None, inplace=False):
        data = self.split_frame(idx, n_random)
        if inplace:
            self.data = data
            return self
        else:
            other = copy(self)
            other.data = data
            return other

    def train_test_split(self, train_idx=None, train_n_random=None):
        train = self.split(train_idx, train_n_random)
        test_idx = list(set(self.data.index) - set(train.data.index))
        test = self.split(test_idx)
        return train, test


    def random_ref_idx(self, n): # ref stand for reference_images
        return np.random.permutation(self.ref_data.index)[:n]

    def _split_frame_by_ref(self, ref_idx): # ref stand for reference_images
        return self.data.loc[self.data[GADAsetFactory.COL_REFERENCE_INDEX].isin(ref_idx)]

    def _random_split_frame_by_ref(self, n): # ref stand for reference_images
        return self._split_frame_by_ref(self.random_ref_idx(n))

    def split_frame_by_ref(self, ref_idx=None, n_random=None):
        assert (ref_idx is None) != (n_random is None)
        if ref_idx is not None:
            return self._split_frame_by_ref(ref_idx)
        else:
            return self._random_split_frame_by_ref(n_random)

    def split_by_ref(self, ref_idx=None, n_random=None, inplace=False):
        data = self.split_frame_by_ref(ref_idx, n_random)
        ref_idx = sorted(set(data[GADAsetFactory.COL_REFERENCE_INDEX]))
        ref_data = self.ref_data.loc[ref_idx]
        if inplace:
            self.data = data
            self.ref_data = ref_data
            return self
        else:
            other = copy(self)
            other.data = data
            other.ref_data = ref_data
            return other

    def train_test_split_by_ref(self, ref_idx=None, n_random=None, specify='train'):
        assert specify in ['train', 'test']
        splitted = self.split_by_ref(ref_idx, n_random)
        other_ref_idx =  list(set(self.ref_data.index) - set(splitted.ref_data.index))
        other = self.split_by_ref(other_ref_idx)
        if specify is 'train':
            train, test = (splitted, other)
        elif specify is 'test':
            train, test = (other, splitted)
        else:
            raise ValueError()
        return train, test



    def print_rows(self):
        for id, row in self.data.iterrows():
            print(f"id:         {id}")
            print(f"Reference:  {row[GADAsetFactory.COL_REFERENCE_NAME]}")
            print(f"Distorted:  {row[GADAsetFactory.COL_DISTORTED_NAME]}")
            print(f"Dist type:  {row[GADAsetFactory.COL_DISTORTION_CATEG]}")
            print(f"Dist level: {row[GADAsetFactory.COL_DISTORTION_LEVEL]}\n")


    def reference_images(self):
        if GADAsetFactory.COL_REFERENCE_IMAGES in self.data.columns:
            try:
                return np.stack(self.data[GADAsetFactory.COL_REFERENCE_IMAGES])
            except:
                # If all images are not with the same dimension, return a list not a tensor
                return list(self.data[GADAsetFactory.COL_REFERENCE_IMAGES])

        #return self.data[COL_REFERENCE_IMAGES]

    def distorted_images(self):
        if GADAsetFactory.COL_DISTORTED_IMAGES in self.data.columns:
            try:
                return np.stack(self.data[GADAsetFactory.COL_DISTORTED_IMAGES])
            except:
                # If all images are not with the same dimension, return a list not a tensor
                return list(self.data[GADAsetFactory.COL_DISTORTED_IMAGES])
        #return self.data[COL_DISTORTED_IMAGES]

    def distortion_types(self) -> np.ndarray:
        return np.expand_dims(np.array(self.data[GADAsetFactory.COL_DISTORTION_CATEG], dtype=np.long), axis=-1)
        #return self.data[COL_DISTORTION_TYPE]

    def distortion_level(self) -> np.ndarray:
        return np.expand_dims(np.array(self.data[GADAsetFactory.COL_DISTORTION_LEVEL], dtype=np.float32), axis=-1)
        #return self.data[COL_DISTORTION_LEVEL]

    def indices(self):
        return np.expand_dims(np.array(self.data.index, dtype=np.int32), axis=-1)
        #return self.data.index

    def ref_indices(self):
        return np.expand_dims(np.array(self.ref_data.index, dtype=np.int32), axis=-1)
        #return self.data.index

    def ref_names(self):
        return np.expand_dims(np.array(self.ref_data[GADAsetFactory.COL_REFERENCE_NAME]), axis=-1)
        #return self.data.index

    def save(self, path):
        dir = '/'.join(path.split('/')[:-1])
        if dir is not '':
            os.makedirs(dir, exist_ok=True)
        import io, pickle
        data_buf = io.BytesIO()
        ref_data_buf = io.BytesIO()
        self.data.to_pickle(data_buf)
        self.ref_data.to_pickle(ref_data_buf)
        pickle.dump({'data': data_buf, 'ref_data': ref_data_buf}, open(path, 'wb'))
        return self

    def load(self, path, inplace=True):
        #self.data = pd.read_pickle(path)
        import pickle
        obj = self if inplace else deepcopy(self)
        d = pickle.load(open(path, 'rb'))
        obj.data = pd.read_pickle( d['data'])
        obj.ref_data = pd.read_pickle( d['ref_data'])
        return obj

    def save_ref_split(self, path):
        dir = '/'.join(path.split('/')[:-1])
        if dir is not '':
            os.makedirs(dir, exist_ok=True)
        f = open(path, 'w')
        for idx in sorted(self.ref_data.index):
            f.write(str(idx) + '\n')
        f.close()

    def load_ref_split(self, path, inplace=False):
        idx = open(path, 'r').read().strip().split('\n')
        idx = [int(id) for id in idx]
        return self.split_by_ref(idx, inplace=inplace)

    def save_split(self, path):
        dir = '/'.join(path.split('/')[:-1])
        if dir is not '':
            os.makedirs(dir, exist_ok=True)
        f = open(path, 'w')
        for idx in sorted(self.data.index):
            f.write(str(idx) + '\n')
        f.close()

    def load_split(self, path, inplace=False):
        idx = open(path, 'r').read().strip().split('\n')
        idx = [int(id) for id in idx]
        return self.split(idx, inplace=inplace)

    def apply_split_file(self, path):
        return self.load_split(path, inplace=True)

    # def __repr__(self):
    #     from tabulate import tabulate
    #     headers = [GADAsetFactory.COL_INDEX] + self.metadata_cols
    #     not_metadata = set(self.data.columns) - set(self.metadata_cols)
    #
    #     ndarrays = [col for col in not_metadata if hasattr(self.data[col][list(self.data[col].keys())[0]], 'shape')]
    #     not_metadata = not_metadata - set(ndarrays)
    #
    #     data = [[id] + [row[col] for col in self.metadata_cols] + [row[col].shape for col in ndarrays] +\
    #             [type(row[col]) for col in not_metadata] for id, row in self.data.iterrows()] # with pandas DataFrame
    #     #data = [[id] + [data_var.data for data_var in self.data.isel(index=id).data_vars.values()] for id in self.data.index] # with xarray
    #     return tabulate(data, headers)
    def __repr__(self):
        all_cols = list(self.data.columns)
        if GADAsetFactory.COL_DISTORTED_IMAGES in all_cols:
            all_cols.remove(GADAsetFactory.COL_DISTORTED_IMAGES)
        if GADAsetFactory.COL_REFERENCE_IMAGES in all_cols:
            all_cols.remove(GADAsetFactory.COL_REFERENCE_IMAGES)
        img_loaded_str = f"\n\nimage_loaded: {self.image_loaded}"

        return f"data:\n{self.data[all_cols].__repr__()}{img_loaded_str}\n\ndata.columns={tuple(self.data.columns)}"




    def generate_ref_splits(self, path, ref_imgs_per_split=5, n_splits=10,
                            train_name='train',
                            test_name='test',
                            first_split_index=1,
                            save=True):
        keys = (train_name, test_name)
        train_test_splits = {i: {k: splt for k, splt in zip(keys, self.train_test_split_by_ref(n_random=ref_imgs_per_split, specify='test'))}
                             for i in range(first_split_index, n_splits+first_split_index)}
        if save:
            for split_id, tt_split in train_test_splits.items():
                tt_split[train_name].save_ref_split(join(path, f'{train_name}_{split_id}'))
                tt_split[test_name].save_ref_split(join(path, f'{test_name}_{split_id}'))
        return train_test_splits

    def load_ref_splits(self, path, train_name='train', test_name='test'):
        import os
        files = os.listdir(path)
        train = {}
        test = {}
        for file in sorted(files):
            if file.startswith(train_name+'_'):
                split_id = '_'.join(file.split('_')[1:])
                try:
                    split_id = int(split_id.split('.')[0])
                    train[split_id] = self.load_ref_split(join(path, file))
                except:
                    pass
            elif file.startswith(test_name + '_'):
                split_id = '_'.join(file.split('_')[1:])
                try:
                    split_id = int(split_id.split('.')[0])
                    test[split_id] = self.load_ref_split(join(path, file))
                except:
                    pass
        assert sorted(train.keys()) == sorted(test.keys())
        return {i: {train_name: train[i], test_name: test[i]} for i in train.keys()}

    def tensor_dataset(s, random_crop_size=None, jpeg_grid=False, verbose=True, device=None):
        from time import time

        if s.image_loaded:
            ref_imgs, dist_imgs = s.reference_images(), s.distorted_images()
        else:
            ref_imgs, dist_imgs = s.load_images(inplace=False, verbose=True)

        if verbose:
            print('Creating torch dataset...')
            now = time()

        from torchvision.transforms import Compose
        transform = Compose([ToTensor(), ToDevice(device)])
        if random_crop_size is not None:
            global_transform = UnisonRandomCrop(random_crop_size, jpeg_grid_alignment=jpeg_grid)
        else:
            global_transform = None

        # if isinstance(imgs[0], np.ndarray) and isinstance(imgs[1], np.ndarray):
        #     # dataset = TensorDatasetGADA(torch.tensor(ref_imgs).to(device),  # ref
        #     #                             torch.tensor(dist_imgs).to(device),  # dist
        #     #                             torch.tensor(s.distortion_types(), dtype=torch.long).to(device),
        #     #                             torch.tensor(s.distortion_level(), dtype=torch.float32).to(device),
        #     #                             torch.tensor(s.indices()).to(device),
        #     #                             img_channel_stack_transform=random_crop)
        #     img_dataset = NumpyArrayDataset(imgs[0], imgs[1], transform_fn=transform, global_transform_fn=global_transform)
        #     meta_dataset = TensorDataset(torch.tensor(s.distortion_types(), dtype=torch.long).to(device),
        #                                  torch.tensor(s.distortion_level(), dtype=torch.float32).to(device),
        #                                  torch.tensor(s.indices()).to(device))
        #     dataset = StackDataset([img_dataset, meta_dataset])
        #
        # else:
        img_dataset = ListOfNumpyArrayDataset(ref_imgs, dist_imgs, transform_fn=transform, global_transform_fn=global_transform)
        meta_dataset = TensorDataset(torch.tensor(s.distortion_types(), dtype=torch.long).to(device),
                                     torch.tensor(s.distortion_level(), dtype=torch.float32).to(device),
                                     torch.tensor(s.indices()).to(device))
        dataset = StackDataset([img_dataset, meta_dataset])



        if verbose:
            print(f'...DONE: Dataset created in {time() - now} seconds')
        return dataset

GADAsetFactory.set_display_options()