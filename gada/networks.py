import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

# FROM: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

###############################################################################
# Helper Functions
###############################################################################

from torch.nn import functional as NN
from numbers import Number

def same_pad(kernel_size, stride):
    return max(kernel_size - stride, 0) // 2


def paddingSame2d(img, next_kernel_size, next_stride, next_dilation=1, mode='replicate', const_value=0, deconv=False):
    stride = (next_stride, next_stride) if isinstance(next_stride, Number) else next_stride
    kernel_size = (next_kernel_size, next_kernel_size) if isinstance(next_kernel_size, Number) else next_kernel_size
    dilation = (next_dilation, next_dilation) if isinstance(next_dilation, Number) else next_dilation
    in_height = img.shape[-2]
    in_width = img.shape[-1]
    pad_h = 0
    pad_w = 0

    # if self.filter_size[0] != in_height: # TODO: does tensorflow make this check?
    if in_height % stride[0] == 0:
        pad_h = max(kernel_size[0] - stride[0], 0)
    else:
        pad_h = max(kernel_size[0] - (in_height % stride[0]), 0)

    # if self.filter_size[1] != in_width: # TODO: does tensorflow make this check?
    if in_width % stride[1] == 0:
        pad_w = max(kernel_size[1] - stride[1], 0)
    else:
        pad_w = max(kernel_size[1] - (in_width % stride[1]), 0)

    if deconv:
        pad_h = dilation[0]*(kernel_size[0]-1) * pad_h
        pad_w = dilation[1]*(kernel_size[1]-1) * pad_w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img = NN.pad(img, pad=[pad_top, pad_bottom, pad_left, pad_right], mode=mode, value=const_value)
    return img

class PaddingSame2d(nn.Module):
    def __init__(self, next_kernel_size, next_stride, next_dilation=1, mode='reflect', const_value=0):
        """
        :param mode: 'constant', 'reflect' or 'replicate'
        :param const_value: fill value for 'constant' padding. Default: 0
        """
        super().__init__()
        self.mode = mode
        self.const_value = const_value
        self.stride = (next_stride, next_stride) if isinstance(next_stride, Number) else next_stride
        self.kernel_size = (next_kernel_size, next_kernel_size) if isinstance(next_kernel_size, Number) else next_kernel_size
        self.dilation = (next_dilation, next_dilation) if isinstance(next_dilation, Number) else next_dilation

    def forward(self, img):
        return paddingSame2d(img, self.kernel_size, self.stride, self.dilation, self.mode, self.const_value)

    def extra_repr(self):
        return f"kernel_size={self.kernel_size}, stride={self.stride}, mode={self.mode}" \
                + (f", const_value={self.const_value}" if self.mode is 'constant' else '')

# class TransposePaddingSame2d(PaddingSame2d):
#     def __init__(self, next_kernel_size, next_stride, next_dilation=1, mode='replicate', const_value=0):
#         """
#         :param mode: 'constant', 'reflect' or 'replicate'
#         :param const_value: fill value for 'constant' padding. Default: 0
#         """
#         super().__init__( next_kernel_size, next_stride, next_dilation, mode, const_value)
#
#     def forward(self, img):
#         return paddingSame2d(img, self.kernel_size, self.stride, self.dilation, self.mode, self.const_value, deconv=True)



class Identity(nn.Module):
    def forward(self, x):
        return x


class UNet(nn.Module):
    """Create a Unet-based generator"""
    @classmethod
    def blocks_builder(cls, first=64, nb_blocks=4, nb_ff_blocks=0):
        """
        :param first: first conv filters - number of output filters for the first convolutional layer.
        :param nb_blocks: number of blocks for UNet models (i.e. number of convolutions and deconvolutions)
        :param nb_ff_blocks: number of fixed-filters blocks, i.e. that doesn't increase the number of filter with respect to the previous
                             layer. Max: nb_blocks-1, Min: 0. Use negative numbers N to automatically compute nb_blocks-N.
        :return: block list that can be passed to the UNet constructor
        """
        if nb_ff_blocks < 0:
            nb_ff_blocks = nb_blocks - nb_ff_blocks
        assert isinstance(nb_ff_blocks, int) and nb_blocks > nb_ff_blocks >= 0
        return [first * 2 ** i for i in range(nb_blocks - nb_ff_blocks)] + [first * 2 ** (nb_blocks - nb_ff_blocks - 1) for i in range(nb_ff_blocks)]


    def __init__(self,
                 blocks=(64, 128, 256, 512),
                 input_channels=3, output_channels=None,
                 norm_layer=nn.BatchNorm2d,  #use_dropout=False,
                 cat_embedding_channels=0,
                 default_stride=2,
                 kernel_size=4,
                 disable_skip_connections=False,
                 use_resize_conv=False,
                 append_layer=None):
        """
        Construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        :param blocks: number of filters for conv/deconv layers in each block.
        :param input_channels: number of channels for the input tensors.
        :param output_channels: number of channels for the output tensors.
        :param norm_layer: normalization layer used in each block.
        :param cat_embedding_channels: number of features concatenated to the input of the decoding architecture. If  not 0, you must
                                       supply in input to the network two tensors, the second one should be an embedding with the
                                       number of channels equals to this parameter, and with spacial dimensions equal to the downsampled
                                       input (each block downsample by 2 --> downsampling factor = 2^(nb_blocks)
        """

        super(UNet, self).__init__()

        #nb_ff_blocks += 1
        output_channels = output_channels if output_channels is not None else input_channels


        strides = []
        filters = []
        for f in blocks:
            if isinstance(f, list) or isinstance(f, tuple):
                filters.append(f[0])
                strides.append(f[1])
            else:
                filters.append(f)
                strides.append(default_stride)

        blocks = filters
        conv_blocks_filters = blocks
        conv_blocks_in = [input_channels] + [f for f in conv_blocks_filters[:-1]]
        deconv_blocks_filters = [output_channels] + conv_blocks_in[1:]

        unet_block = None
        for i, (conv_in, conv_f, deconv_f, stride) in enumerate(zip(conv_blocks_in[::-1], conv_blocks_filters[::-1], deconv_blocks_filters[::-1], strides[::-1])):
            extra_channels = cat_embedding_channels if i==0 else 0
            unet_block = UNetSkipConnBlock(conv_in, conv_f, deconv_f, extra_deconv_input_channels=extra_channels,
                                           norm_layer=norm_layer, innermost=(i==0), outermost=(i==len(blocks)-1), submodule=unet_block,
                                           kernel_size=kernel_size, stride=stride, pad_mode='replicate',
                                           disable_skip_connection=disable_skip_connections,
                                           use_resize_conv=use_resize_conv)

        self.model = unet_block
        self.appended = append_layer

        self.blocks = blocks
        self.strides = strides
        self.disable_skip_connections = disable_skip_connections

    def compute_encoded_shape(self, input_tensor_or_shape):
        shape = input_tensor_or_shape
        if isinstance(input_tensor_or_shape, torch.Tensor):
            shape = input_tensor_or_shape.shape
        if len(shape) < 3:
            raise ValueError("The shape of the tensor should have 3 or 4 dimensions: ([batch_index,] chanels, Y, X)")
        c=-3; h=-2; w=-1
        out = list(shape)
        for stride in self.strides:
            out[w]//=stride
            out[h] //= stride
        out[c] = self.blocks[-1]
        return out

    def forward(self, *input):
        """Standard forward"""
        res = self.model(*input)
        return res if self.appended is None else self.appended(res)



class UNetSkipConnBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, input_channels, conv_filters, deconv_filters=None, extra_deconv_input_channels=0,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 stride=2, kernel_size=4, pad_mode='reflect', deconv_pad_mode=None, pad_value=0, disable_skip_connection=False,
                 use_resize_conv=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            :param deconv_filters (int) -- the number of filters in the outer conv layer
            :param conv_filters (int) -- the number of filters in the inner conv layer
            :param input_channels (int) -- the number of channels in input images/features
            :param submodule (UnetSkipConnectionBlock) -- previously defined submodules
            :param outermost (bool)    -- if this module is the outermost module
            :param innermost (bool)    -- if this module is the innermost module
            :param norm_layer          -- normalization layer
            :param user_dropout (bool) -- if use dropout layers.

            :param stride:
            :param kernel_size:
            :param pad_mode: 'constant', 'reflect' or 'replicate'
            :param pad_value: fill value for 'constant' padding. Default: 0
        """
        super(UNetSkipConnBlock, self).__init__()

        self.outermost = outermost
        self.innermost = innermost
        self.disable_skip_connection = disable_skip_connection
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        deconv_filters = deconv_filters if deconv_filters is not None else input_channels

        self.input_channels = input_channels
        self.output_channels = deconv_filters

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(conv_filters)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(deconv_filters)
        skip = 1 if disable_skip_connection else 2

        downconv = nn.Conv2d(input_channels, conv_filters, kernel_size=kernel_size, stride=stride, bias=use_bias)

        # From: https://pytorch.org/docs/stable/nn.html --> ConvTranspose2d
        # !!! The padding argument effectively adds "dilation * (kernel_size - 1) - padding" amount of zero padding to both sizes of the input
        # --> We want to remove the padding automatically added when padding=0, i.e. "dilation * (kernel_size - 1)"
        dilation=1
        transpose_pad = dilation * (kernel_size-1) # disable transpose-padding
        deconv_pad_mode = pad_mode if deconv_pad_mode is None else deconv_pad_mode
        if use_resize_conv:
            transpose_padding_same = PaddingSame2d(kernel_size, 1, mode=deconv_pad_mode, const_value=pad_value)
        else:
            transpose_padding_same = PaddingSame2d(kernel_size, stride, mode=deconv_pad_mode, const_value=pad_value)


        padding_same = PaddingSame2d(kernel_size, stride, mode=pad_mode, const_value=pad_value)

        conv_in_chs = (conv_filters+extra_deconv_input_channels) if innermost else  (conv_filters*skip+extra_deconv_input_channels)
        if use_resize_conv:
            upconv = nn.ConvTranspose2d(conv_in_chs, deconv_filters, kernel_size, stride, padding=0, bias=use_bias)
        else:
            upconv = nn.ConvTranspose2d(conv_in_chs, deconv_filters, kernel_size, stride, padding=transpose_pad, bias=use_bias)
        upsample = nn.UpsamplingNearest2d(scale_factor=stride)
        if outermost:
            down = [padding_same, downconv, downrelu]
            up = [upconv]
        else:  # innermost or middle
            down = [padding_same, downconv, downnorm, downrelu]
            up = [upconv, upnorm, uprelu]

        if transpose_padding_same is not None:
            up = [transpose_padding_same] + up
        if use_resize_conv:
            up = [upsample] + up
        if use_dropout and not innermost and not outermost:
            up += [nn.Dropout(0.5)]

        self.down = nn.Sequential(*down)
        self.sub = submodule
        self.up = nn.Sequential(*up)

    def forward(self, *x):
        x = list(x)
        aux_embedding = x[1] if len(x) > 1 else None
        x_in = x[0]

        out = self.down(x_in)
        if self.sub is not None:
            x[0] = out
            out = self.sub(*x)

        if self.innermost and aux_embedding is not None:
            x_up = self.up(torch.cat((out, aux_embedding), 1))
        else:
            x_up = self.up(out)

        if self.outermost or self.disable_skip_connection:
            return x_up
        else:   # add skip connections
            return torch.cat((x_up, x_in), 1)









class FullyConvFeatExtractor(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_size, blocks=(64, 128, 256), input_channels=3, disable_full_size_conv=True, norm_layer=nn.BatchNorm2d,
                 kernel_size=4, default_stride=2,
                 padding='same', same_pad_mode='constant', same_pad_const=0, last_padding=0, last_stride=1,
                 append_layer=None):
        """
        Create a feature extractor network that should be used before a fully convolutional FC-like layer.
        Using 3 blocks it will create 4 convolutions: the first 3 are the ones specified by the parameters, the last one
        will replicate the number F of filters of the last convolution and reduce a target image to a BSxFx1x1 tensor,
        where BS is the batch size. For bigger images the tensor could have more elements in the last two dimensions, and
        a pooling layer or a mean can be used to reduce to a BSxFx1x1 again.

        :param input_size: spatial input size (tensor width/height, width should be equal to height)
        :param blocks: sequence of number, each one is the number of filters of a convolutional layer.
                       You can also pass a sequence of couples where for each one the first number is the number of filters,
                        the second one is the stride for that convolution (i.e. the downsampling factor).
        :param input_channels: number of channels in the input tensor (3 for rgb images).
        :param disable_full_size_conv: if False, an additional convolution with kernel_size==input_size will be added so that inputs with the
                               selected size will have an output tensor of shape of (?, blocks[-1], 1, 1)
        :param norm_layer:
        :param kernel_size:
        :param default_stride:
        :param padding:
        :param last_padding:
        """
        super(FullyConvFeatExtractor, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        strides = []
        filters = []
        for f in blocks:
            if isinstance(f, list) or isinstance(f, tuple):
                filters.append(f[0])
                strides.append(f[1])
            else:
                filters.append(f)
                strides.append(default_stride)


        #input_size = (input_size, input_size) if isinstance(input_size, int) else input_size

        sequence = []
        in_chanels = input_channels

        for i, (f, s) in enumerate(zip(filters, strides)):
            if padding is 'same':
                pad = 0
                sequence += [PaddingSame2d(kernel_size, s, next_dilation=1, mode=same_pad_mode, const_value=same_pad_const)]
            else:
                pad = padding
            sequence += [nn.Conv2d(in_chanels, f, kernel_size=kernel_size, stride=s, padding=pad, bias=use_bias)]
            if i != 0: # not for the first block
                sequence += [norm_layer(f)]
            sequence += [nn.LeakyReLU(0.2, True)]
            in_chanels = f


        if not disable_full_size_conv:
            last_kernel_size = input_size
            for s in strides:
                last_kernel_size //= s
            sequence += [nn.Conv2d(in_chanels, filters[-1],
                                   kernel_size=last_kernel_size, stride=last_stride, padding=last_padding, bias=use_bias),
                         norm_layer(filters[-1]),
                         nn.LeakyReLU(0.2, True)]

        self.output_channels = filters[-1]
        self.input_channels = input_channels

        self.model = nn.Sequential(*sequence)
        self.appended = append_layer

    def forward(self, input):
        """Standard forward"""
        res = self.model(input)
        return res if self.appended is None else self.appended(res)




class FullyConvFC(nn.Module):
    def __init__(self, input_channels, output_channels, reduce='mean', append_layer=None):
        """Construct a discriminator to be used on the output of the NLayerFeatExtractor
        Parameters:
            input_channels (int)  -- the number of channels in input images
        """
        super().__init__()
        self.model = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)  # output 1 channel prediction map
        if append_layer is not None:
            self.model = nn.Sequential(self.model, append_layer)
        self.reduce_fn = lambda t: t
        if reduce is 'mean':
            self.reduce_fn = lambda t: torch.mean(t, dim=(-1, -2), keepdim=False)

    def forward(self, input):
        """Standard forward."""
        return self.reduce_fn(self.model(input))

class FeatureDiscriminator(FullyConvFC):
    def __init__(self, input_nc, reduce='mean', append_layer=None):
        """Construct a discriminator to be used on the output of the NLayerFeatExtractor
        Parameters:
            input_nc (int)  -- the number of channels in input images
        """
        super().__init__(input_nc, 1, reduce, append_layer)




#%%
if __name__ == "__main__":
    unet = UNet(blocks=(64, 128, 256), input_channels=3, cat_embedding_channels=20)
    print(unet)
    unet = UNet(UNet.blocks_builder(64, 3, 0), input_channels=3, cat_embedding_channels=20)
    print(unet)

    from fastorch.summary import summary

    F = FullyConvFeatExtractor(input_size=128, blocks=(64, 128, 256), input_channels=3, last_padding=0)
    D = FullyConvFC(F.output_channels, 1)
    C = FullyConvFC(F.output_channels, 10)
    E = FullyConvFC(F.output_channels, 1)

    print(unet)
    summary(nn.Sequential(F, D), torch.zeros(2, 3, 128, 128), device='cpu', print_params=True);


#
# class UNetSkipConnBlock_OLD(nn.Module):
#     """Defines the Unet submodule with skip connection.
#         X -------------------identity----------------------
#         |-- downsampling -- |submodule| -- upsampling --|
#     """
#
#     def __init__(self, outer_nc, inner_nc, input_nc=None,
#                  submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
#         """Construct a Unet submodule with skip connections.
#
#         Parameters:
#             outer_nc (int) -- the number of filters in the outer conv layer
#             inner_nc (int) -- the number of filters in the inner conv layer
#             input_nc (int) -- the number of channels in input images/features
#             submodule (UnetSkipConnectionBlock) -- previously defined submodules
#             outermost (bool)    -- if this module is the outermost module
#             innermost (bool)    -- if this module is the innermost module
#             norm_layer          -- normalization layer
#             user_dropout (bool) -- if use dropout layers.
#         """
#         super(UNetSkipConnBlock, self).__init__()
#         self.outermost = outermost
#         self.innermost = innermost
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#         if input_nc is None:
#             input_nc = outer_nc
#         downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
#                              stride=2, padding=1, bias=use_bias)
#         downrelu = nn.LeakyReLU(0.2, True)
#         downnorm = norm_layer(inner_nc)
#         uprelu = nn.ReLU(True)
#         upnorm = norm_layer(outer_nc)
#
#         if outermost:
#             upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1)
#             down = [downconv]
#             up = [uprelu, upconv, nn.Tanh()]
#         elif innermost:
#             upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downrelu, downconv]
#             up = [uprelu, upconv, upnorm]
#         else:
#             upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downrelu, downconv, downnorm]
#             up = [uprelu, upconv, upnorm]
#
#             if use_dropout:
#                 up += [nn.Dropout(0.5)]
#
#         self.down = nn.Sequential(*down)
#         self.sub = submodule
#         self.up = nn.Sequential(*up)
#
#     def forward(self, *x):
#         aux_embedding = x[1] if len(x) > 1 else None
#         x_in = x[0]
#
#         x_down = self.down(x_in)
#         if self.sub is not None:
#             x_down = self.sub(x_down)
#
#         if self.innermost and aux_embedding is not None:
#             x_up = self.up(torch.cat((x_down, aux_embedding), 1))
#         else:
#             x_up = self.up(x_down)
#
#         if self.outermost:
#             return x_up
#         else:   # add skip connections
#             return torch.cat((x_up, x_in), 1)
#
