import numbers

from torch.utils.data.dataset import Dataset
from torch.nn import functional as NNF


class StackDataset(Dataset):
    def __init__(self, datasets):
        assert all(len(datasets[0]) == len(d) for d in datasets)
        self.datasets = datasets

    def __getitem__(self, index):
        #return tuple(d[index] for d in self.datasets)
        return tuple(col for d in self.datasets for col in d[index])

    def __len__(self):
        return len(self.datasets[0])


class ArrayListDataset(Dataset):
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


class ArrayDataset(Dataset):

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
        w_dim = -1
        h_dim = -2
        h = img_tensor.shape[h_dim]
        w = img_tensor.shape[w_dim]

        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, (h - th) // grid_alignment) * grid_alignment
        j = random.randint(0, (w - tw) // grid_alignment) * grid_alignment

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
        return img_tensor[:, i:i + h, j:j + w]
        # return F.crop(img_tensor, i, j, h, w)

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
            img_tensors_cropped.append(joined_img_tensor_cropped[ch_st:ch_st + ch])
            ch_st += ch
        return tuple(img_tensors_cropped)



