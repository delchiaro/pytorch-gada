import numbers
import random

import torch
from abc import ABC, abstractmethod
import numpy as np
from torch.utils.data import TensorDataset, Dataset
from torchvision.transforms import RandomCrop, Compose
from torch.nn import functional as NNF


class TensorRandomCrop(object):
    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img_tensor, output_size):
        w_dim=-1
        h_dim=-2
        h = img_tensor.shape[h_dim]
        w = img_tensor.shape[w_dim]

        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img_tensor):
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

        i, j, h, w = self.get_params(img_tensor, self.size)
        return img_tensor[:, i:i+h, j:j+w]
        #return F.crop(img_tensor, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class TensorGADAset(TensorDataset):
    def __init__(self, ref_img_tensor, dist_img_tensor, dist_type_tensor, dist_lvl_tensor, indices_tensor,
                 img_transform=None, ref_img_transform=None, dist_img_transform=None):
        super().__init__(*[ref_img_tensor, dist_img_tensor, dist_type_tensor, dist_lvl_tensor, indices_tensor])
        self.img_transform = img_transform
        self.ref_img_transform = ref_img_transform
        self.dist_img_transform = dist_img_transform

    def __getitem__(self, index):
        ref_img, dist_img, dist_type, dist_level, indices = super(TensorGADAset, self).__getitem__(index)
        if self.img_transform is not None:
            ref_img = self.img_transform(ref_img)
            dist_img = self.img_transform(dist_img)
        if self.ref_img_transform is not None:
            ref_img = self.ref_img_transform(ref_img)
        if self.dist_img_transform is not None:
            dist_img = self.dist_img_transform(dist_img)
        return ref_img, dist_img, dist_type, dist_level, indices



class GADAsetFactory(ABC):
    COL_INDEX = 'index'
    COL_DISTORTED_NAME = 'distorted_name'
    COL_REFERENCE_NAME = 'reference_name'
    COL_DISTORTION_TYPE = 'distortion_type'
    COL_DISTORTION_LEVEL = 'distortion_level'
    COLS_METADATA = [COL_DISTORTED_NAME, COL_REFERENCE_NAME, COL_DISTORTION_TYPE, COL_DISTORTION_LEVEL]
    COL_DISTORTED_IMAGES = 'distorted_images'
    COL_REFERENCE_IMAGES = 'reference_images'

    def __init__(self):
        self._image_loaded = False

    @abstractmethod
    def reference_images(self) -> np.ndarray:
        pass

    @abstractmethod
    def distorted_images(self) -> np.ndarray:
        pass

    @abstractmethod
    def distortion_types(self) -> np.ndarray:
        pass

    @abstractmethod
    def distortion_level(self) -> np.ndarray:
        pass

    @abstractmethod
    def indices(self) -> np.ndarray:
        pass


    def load_images(self, inplace=True, verbose=False, **kwargs):
        self._image_loaded=True
        return self._load_images(inplace, verbose, **kwargs)

    @abstractmethod
    def _load_images(self, inplace=True, verbose=False, **kwargs):
        pass

    def drop_images(self, **kwargs):
        self._image_loaded = False
        return self._drop_images(**kwargs)
    @abstractmethod
    def _drop_images(self, **kwargs):
        pass

    @property
    def image_loaded(self):
        return self._image_loaded

    def tensor_dataset(s, random_crop_size=None, verbose=True, device=None):
        from time import time

        if s.image_loaded:
            imgs = s.reference_images(), s.distorted_images()
        else:
            imgs = s.load_images(inplace=False, verbose=True)

        if verbose:
            print('Creating torch dataset...')
            now = time()

        random_crop = TensorRandomCrop(random_crop_size) if random_crop_size is not None else None
        dataset = TensorGADAset(torch.tensor(imgs[0], dtype=torch.uint8).to(device),
                                torch.tensor(imgs[1], dtype=torch.uint8).to(device),
                                torch.tensor(s.distortion_types(), dtype=torch.uint8).to(device),
                                torch.tensor(s.distortion_level(), dtype=torch.uint8).to(device),
                                torch.tensor(s.indices(), dtype=torch.uint8).to(device),
                                img_transform=random_crop)

        if verbose:
            print(f'...DONE: Dataset created in {time() - now} seconds')
        return dataset

