from typing import List
from torch import nn
from collections.abc import Iterable

from torch.nn.modules.conv import _ConvNd


import fastorch.layers as flayers

def conv_out_size(input_size, filter_size, stride, padding):
    return (input_size - filter_size + 2 * padding) / stride + 1


def convNd_out_shape(conv: _ConvNd, in_shape, Nd=None):
    if Nd is None:
        Nd = len(in_shape) - 1
    else:
        assert len(in_shape) == Nd + 1
    k_size = conv.kernel_size if isinstance(conv.kernel_size, Iterable) else [conv.kernel_size] * Nd
    stride = conv.stride if isinstance(conv.stride, Iterable) else [conv.stride] * Nd
    padding = conv.padding if isinstance(conv.padding, Iterable) else [conv.padding] * Nd
    return [conv_out_size(in_shape[k], k_size[k], stride[k], padding[k]) for k in range(Nd)] + [conv.out_channels]


#
# def conv1d_out_shape(conv1d: nn.Conv2d, in_shape):
#     return conv_out_shape(conv1d, in_shape, 1)
#
#
# def conv2d_out_shape(conv2d: nn.Conv2d, in_shape):
#     return conv_out_shape(conv2d, in_shape, 2)
#
#
# def conv3d_out_shape(conv3d: nn.Conv2d, in_shape):
#     return conv_out_shape(conv3d, in_shape, 3)


def out_shapes(modules: List[nn.Module], in_shape):
    modules = [modules] if isinstance(modules, nn.Module) else modules
    shapes = []
    for m in modules:
        in_shape = out_shape(m, in_shape)
        shapes.append(in_shape)
    return shapes


def out_shape(module: nn.Module, in_shape=None):
    if hasattr(module, 'kernel_size') and hasattr(module, 'stride') and hasattr(module, 'padding'):
        out = convNd_out_shape(module, in_shape)
    elif isinstance(module, nn.Linear) or hasattr(module, 'out_features'):
        out = module.out_features
    else:
        raise TypeError()
    return out


def wshape(module: nn.Module, in_shape=None):
    return out_shape(module, in_shape), module


def fastModules(layers: list, input_shape=None, default_lrelu_leak=.2, default_dropout_rate=0.5, default_bias=True) \
        -> (List[nn.Module], int):
    modules = []
    next_bias = default_bias
    next_input_shape = input_shape
    i = 0
    while i < len(layers):
        if isinstance(layers[i], str):
            if layers[i] == 'dropout':
                modules.append(nn.Dropout(default_dropout_rate))

            if layers[i] == 'relu':
                modules.append(nn.ReLU())

            if layers[i] == 'softmax':
                modules.append(nn.Softmax())

            if layers[i] == 'lrelu':
                modules.append(nn.LeakyReLU(default_lrelu_leak))

        elif isinstance(layers[i], int):
            modules.append(nn.Linear(next_input_shape, layers[i], next_bias))
            next_bias = default_bias
            next_input_shape = layers[i]

        elif isinstance(layers[i], float):
            modules.append(nn.Dropout(layers[i]))

        elif isinstance(layers[i], flayers.FastModule):
            modules.append(layers[i])
            if next_input_shape is not None:
                next_input_shape = layers[i].out_shape(next_input_shape)

        elif isinstance(layers[i], nn.Module):
            modules.append(layers[i])
            if next_input_shape is not None:
                next_input_shape = out_shape(layers[i], next_input_shape)

        elif (isinstance(layers[i], tuple) or isinstance(layers[i], list)) and len(layers[i] == 2):
            modules.append(layers[i][0])
            next_input_shape = layers[i][1]

        i += 1

    #module = nn.Sequential(*modules)
    return modules, next_input_shape


def fastSequential(layers: list, input_shape=None, default_lrelu_leak=.2, default_dropout_rate=0.5, default_bias=True) -> \
        (nn.Sequential, int):
    modules, shape = fastModules(layers, input_shape, default_lrelu_leak, default_dropout_rate, default_bias)
    return nn.Sequential(*modules), out_shape
