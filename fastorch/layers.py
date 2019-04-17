from abc import ABC, abstractmethod
import torch
from torch import nn
from .utils import conv_out_size


class FastModule(torch.nn.Module, ABC):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    @property
    @abstractmethod
    def out_channels(self):
        pass

    @abstractmethod
    def out_shape(self, in_shape=None):
        pass


class Conv2d(torch.nn.Conv2d, FastModule):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__(in_channels, out_channels, kernel_size)
        self._out_channels=in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def out_shape(self, in_shape=None):
        assert len(in_shape) == 3
        return (conv_out_size(in_shape[0], self.kernel_size[0], self.stride[0], self.padding[0]),
                conv_out_size(in_shape[1], self.kernel_size[1], self.stride[1], self.padding[1]),
                self.out_channels)


class Linear(torch.nn.Linear, FastModule):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)

    def in_shape(self):
        return self.in_features

    def out_shape(self, in_shape=None):
        return self.out_features

    @property
    def out_channels(self):
        return self.out_features


# class Sequential(torch.nn.Sequential, FastModule):
#     def __init__(self, modules, in_shape=None, out_shape=None):
#         super().__init__(modules)
#         self._out_shapes = {}
#         if in_shape is not None and out_shape is not None:
#             self._out_shapes[in_shape] = out_shape
#
#     def out_shape(self, in_shape=None):
#         if in_shape not in self._out_shapes.keys():
#             self._out_shapes[in_shape] = out_shapes(self.modules())[-1]
#         return self._out_shapes[in_shape]
