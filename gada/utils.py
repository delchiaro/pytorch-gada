import torch
from torch import nn

def conv_out_size(input_size, filter_size, stride, padding):
    return (input_size - filter_size + 2 * padding) / stride + 1


def conv2d_out_shape(input_size, conv2d: nn.Conv2d):
    return [conv2d.out_channels,
            conv_out_size(input_size[0], conv2d.kernel_size[0], conv2d.stride[0], conv2d.padding[0]),
            conv_out_size(input_size[1], conv2d.kernel_size[1], conv2d.stride[1], conv2d.padding[1])]



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def to_float32(*tensors):
    return tuple(t.float() for t in tensors)

def to_device(*tensors, device):
    return tuple(t.to(device) for t in tensors)

def index_to_1hot(indices, nb_elements):
    return torch.diag(torch.ones((nb_elements,), device=indices.device))[indices]
