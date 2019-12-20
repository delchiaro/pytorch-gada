import torch
from torch import nn
from numbers import Number
from fastorch import fastSequential
from fastorch.summary import summary
from utils import index_to_1hot
import torch.distributions as tdist

import torch.nn.functional as NN

DEFAULT_LEAK = .2


def NP(tensor):
    return tensor.detach().cpu().numpy()


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def padding_same(filter_size, stride):
    return max(filter_size - stride, 0) // 2


class PaddingSame2d(nn.Module):
    def __init__(self, next_kernel_size, next_stride, mode='replicate', const_value=0):
        """
        :param mode: 'constant', 'reflect' or 'replicate'
        :param const_value: fill value for 'constant' padding. Default: 0
        """
        super().__init__()
        self.mode = mode
        self.const_value = const_value
        if isinstance(next_stride, Number): next_stride = (next_stride, next_stride)
        if isinstance(next_kernel_size, Number): next_kernel_size = (next_kernel_size, next_kernel_size)
        self.stride = next_stride
        self.kernel_size = next_kernel_size

    def forward(self, img):
        in_height = img.shape[-2]
        in_width = img.shape[-1]
        pad_h = 0
        pad_w = 0

        # if self.filter_size[0] != in_height: # TODO: does tensorflow make this check?
        if in_height % self.stride[0] == 0:
            pad_h = max(self.kernel_size[0] - self.stride[0], 0)
        else:
            pad_h = max(self.kernel_size[0] - (in_height % self.stride[0]), 0)

        # if self.filter_size[1] != in_width: # TODO: does tensorflow make this check?
        if in_width % self.stride[1] == 0:
            pad_w = max(self.kernel_size[1] - self.stride[1], 0)
        else:
            pad_w = max(self.kernel_size[1] - (in_width % self.stride[1]), 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        img = NN.pad(img, pad=[pad_top, pad_bottom, pad_left, pad_right], mode=self.mode, value=self.const_value)
        return img

    def extra_repr(self):
        return f"kernel_size={self.kernel_size}, stride={self.stride}, mode={self.mode}" \
               + (f", const_value={self.const_value}" if self.mode is 'constant' else '')


def conv_block(input_channels, nb_filters, downsampling=2, kernel_size=4, batchnorm=True, leak=DEFAULT_LEAK):
    padding = padding_same(kernel_size, downsampling)
    block = [nn.Conv2d(input_channels, nb_filters, kernel_size, stride=downsampling, padding=padding),
             nn.LeakyReLU(leak) if leak is not None else nn.ReLU()]
    if batchnorm:
        block += [nn.BatchNorm2d(nb_filters)]
    return block


def deconv_block(input_channels, nb_filters, upsampling=2, kernel_size=3, output=None, leak=DEFAULT_LEAK):
    block = []
    if upsampling > 1:
        block += [nn.Upsample(scale_factor=upsampling)]
    block += [nn.ReplicationPad2d(padding=padding_same(kernel_size, stride=1)),
              nn.Conv2d(input_channels, nb_filters, kernel_size=kernel_size, stride=1, padding=0)]
    if output is True:
        block += [nn.Sigmoid()]  # todo: remove sigmoid activation from here?
    else:
        block += [nn.LeakyReLU(leak) if leak is not None else nn.ReLU(), nn.BatchNorm2d(nb_filters)]
    return block


class Encoder(nn.Module):
    def __init__(self, in_channels=3, downsampling_filters=(64, 64, 128, 128), output_batchnorm=False):
        super().__init__()
        self.filters = downsampling_filters
        in_chs = [in_channels] + list(downsampling_filters[:-1])
        self.blocks = [conv_block(in_ch, f, batchnorm=True) for in_ch, f in zip(in_chs[:-1], downsampling_filters[:-1])]
        self.blocks += [conv_block(in_chs[-1], downsampling_filters[-1], batchnorm=output_batchnorm)]
        self.blocks = nn.Sequential(*[nn.Sequential(*block) for block in self.blocks])
        self.skip_out = None

    def out_channels(self):
        return self.filters[-1]

    def forward(self, x):
        outs = []
        for conv in self.blocks:
            x = conv(x)
            outs.append(x)
        self.skip_out = outs[:-1]
        return outs[-1]


class Decoder(nn.Module):
    def __init__(self, input_channels, upsampling_filters=(128, 64, 64, 3), extra_filters=None, skip_connections=True):
        super().__init__()
        self.filters = upsampling_filters
        self.skip_connections = skip_connections
        mul = 2 if skip_connections else 1

        # compute channels w or wo skip-connections (from second to last deconv)
        in_chs = [input_channels] + [flt * mul for flt in upsampling_filters[:-1]]

        self.extra_blocks = None
        self.blocks = [deconv_block(in_ch, f) for in_ch, f in zip(in_chs[:-1], upsampling_filters[:-1])]

        if extra_filters is None or len(extra_filters) == 0:
            self.blocks += [deconv_block(in_chs[-1], upsampling_filters[-1], output=True)]
        else:
            self.extra_blocks = [deconv_block(in_ch, f, upsampling=1) for in_ch, f in
                                 zip(in_chs[:-1], extra_filters[:-1])]
            self.extra_blocks += [deconv_block(in_chs[-1], extra_filters[-1], upsampling=1, output=True)]

        self.blocks = nn.Sequential(*[nn.Sequential(*block) for block in self.blocks])
        if self.extra_blocks is not None:
            self.extra_blocks = nn.Sequential(*[nn.Sequential(*block) for block in self.extra_blocks])

    def out_channels(self):
        return self.filters[-1]

    def forward(self, enc_out, *skip_connection_inputs):
        out = self.blocks[0](enc_out)
        if self.skip_connections:
            for deconv, conv_emb in zip(self.blocks[1:], skip_connection_inputs):
                out = deconv(torch.cat((out, conv_emb), dim=1))
        else:
            for deconv in self.blocks[1:]:
                out = deconv(out)

        if self.extra_blocks is not None:
            out = self.extra_blocks(out)
        return out




class Generator(nn.Module):
    def __init__(self, nb_dist_categs, z_noise_dim, filters=(64, 64, 128, 128),
                 dlevel_emb_size=32, dcateg_emb_size=32, input_channels=3):
        super().__init__()
        self.z_dim = z_noise_dim
        self.z_noise_distribution = tdist.Normal(0, 0.1)  # mean=0, stddev=0.1

        self.nb_dist_categs = nb_dist_categs
        self.dlevel_in_emb_size = nb_dist_categs
        self.dlevel_emb_size = dlevel_emb_size
        self.dcateg_emb_size = dcateg_emb_size
        self.input_channels = input_channels

        self.dlevel_in_encoder, _ = fastSequential([self.dlevel_in_emb_size, 'relu',
                                                    self.dlevel_in_emb_size, 'relu'], input_shape=1)

        self.dlevel_encoder, _ = fastSequential([dlevel_emb_size, 'relu',
                                                 dlevel_emb_size, 'relu'], input_shape=1)

        self.dcateg_encoder, _ = fastSequential([dlevel_emb_size, 'relu',
                                                 dcateg_emb_size, 'relu'], input_shape=self.nb_dist_categs)

        self.encoder = Encoder(in_channels=input_channels + nb_dist_categs * 2, downsampling_filters=filters)

        self.decoder = Decoder(input_channels=filters[-1] + z_noise_dim + dlevel_emb_size + dcateg_emb_size,
                               upsampling_filters=list(filters[:-1][::-1]) + [input_channels],
                               extra_filters=None)

    def summary(self, crop_size=128, print_params=False):
        print("\nEncoder: ")
        x = summary(self.encoder, torch.zeros(2, self.input_channels + self.nb_dist_categs * 2, crop_size, crop_size),
                    print_params=print_params)
        print("\nDecoder: ")

        decoder_input = torch.zeros((2, x.shape[
            1] + self.z_dim + self.dlevel_emb_size + self.dcateg_emb_size, x.shape[2],
                                     x.shape[3]))
        decoder_input = (decoder_input, *(self.encoder.skip_out[::-1])) if self.decoder.skip_connections else x
        summary(self.decoder, decoder_input, print_params=print_params)

    def forward(self, img, dist_categ, dist_level):
        dcateg_onehot = index_to_1hot(dist_categ[:, 0].long(), self.nb_dist_categs).float()
        dlevel_input_emb = self.dlevel_in_encoder(dist_level)
        dcateg_dlevel_input_emb = torch.cat((dcateg_onehot, dlevel_input_emb), dim=1)
        dcateg_dlevel_input_emb = dcateg_dlevel_input_emb.unsqueeze(-1).unsqueeze(-1)
        dcateg_dlevel_input_emb = dcateg_dlevel_input_emb.expand(-1, -1, img.shape[2], img.shape[3])

        x = torch.cat((img, dcateg_dlevel_input_emb), dim=1)
        # x = img

        # Encoding the input image (concatenated with an embedding of distortion level/category)
        # enc1, enc2, enc3 = self.encoder(x)
        enc_out = self.encoder(x)
        enc_skip_out = self.encoder.skip_out

        # Embedding distortion level and category
        d_level_emb_out = self.dlevel_encoder(dist_level)
        d_categ_emb_out = self.dcateg_encoder(dcateg_onehot)
        d_categ_level_emb_out = torch.cat((d_categ_emb_out, d_level_emb_out), dim=1)
        d_categ_level_emb_out = d_categ_level_emb_out.unsqueeze(-1).unsqueeze(-1)
        d_categ_level_emb_out = d_categ_level_emb_out.expand(-1, -1, enc_out.shape[2], enc_out.shape[3])

        # Preparing decoder input using the embedding of distortion level/category
        # concatenated to the encoder output and a noise vector.
        sample_shape = (enc_out.shape[0], self.z_dim, enc_out.shape[2], enc_out.shape[3])
        z_noise = self.z_noise_distribution.sample(sample_shape).to(self.device)
        decoder_input = torch.cat((enc_out, d_categ_level_emb_out, z_noise), dim=1)

        # Decoding
        decoded = self.decoder(decoder_input, *(enc_skip_out[::-1]))
        return decoded

    @staticmethod
    def loss(generated_dist_img, real_dist_img):
        return torch.mean(torch.abs(real_dist_img - generated_dist_img))


class FeatureExtractor(nn.Module):
    def __init__(self, out_features=1024, filters=(64, 128, 128), in_channels=3, last_kernel_size=16, leak=DEFAULT_LEAK):
        super().__init__()
        self.filters = filters
        in_chs = [in_channels] + list(filters)[:-1]

        layers = []
        for i, (in_ch, f) in enumerate(zip(in_chs, filters)):
            layers.append(PaddingSame2d(next_kernel_size=4, next_stride=2, mode='replicate'))
            layers.append(nn.Conv2d(in_ch, f, kernel_size=4, stride=2))
            if i > 0:
                layers.append(nn.BatchNorm2d(f))
            layers.append(nn.LeakyReLU(leak) if leak is not None else nn.ReLU())

        layers.append(nn.Conv2d(filters[-1], out_features, kernel_size=last_kernel_size, stride=last_kernel_size))
        layers.append(nn.BatchNorm2d(out_features))
        layers.append(nn.LeakyReLU(leak) if leak is not None else nn.ReLU())

        self.net = nn.Sequential(*layers)
        #
        # self.net = nn.Sequential(PaddedSameConv2d(in_ch, 64, kernel_size=4, stride=2), nn.LeakyReLU(leak),
        #                          PaddedSameConv2d(64, 128, kernel_size=4, stride=2), nn.BatchNorm2d(128), nn.LeakyReLU(leak),
        #                          PaddedSameConv2d(128, 128, kernel_size=4, stride=2), nn.BatchNorm2d(128), nn.LeakyReLU(leak),
        #                          nn.Conv2d(128, out_features, kernel_size=16, stride=1),
        #                          nn.BatchNorm2d(out_features),
        #                          nn.LeakyReLU(leak))

        # self.net = nn.Sequential(PaddedSameConv2d(in_ch, 64, kernel_size=4, stride=2), nn.LeakyReLU(leak),
        #                          PaddedSameConv2d(64, 128, kernel_size=4, stride=2), nn.BatchNorm2d(128), nn.LeakyReLU(leak),
        #                          PaddedSameConv2d(128, 128, kernel_size=4, stride=2), nn.BatchNorm2d(128), nn.LeakyReLU(leak),
        #                          nn.Conv2d(128, out_features, kernel_size=16, stride=1),
        #                          nn.BatchNorm2d(out_features),
        #                          nn.LeakyReLU(leak))

    def summary(self, crop_size=128, in_channels=3, print_params=False):
        summary(self, input_size=(in_channels, crop_size, crop_size), print_params=print_params)

    def forward(self, x):
        return self.net(x)




class Discriminator(nn.Module):
    def __init__(self, in_features=1024, device=None):
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.net = nn.Sequential(nn.Conv2d(in_features, 1, kernel_size=1, stride=1, padding=padding_same(1, 1)),
                                 nn.Sigmoid())

    def forward(self, x):
        return torch.mean(self.net(x), dim=[2, 3], keepdim=False)
        # return torch.squeeze(torch.squeeze(self.net(x), dim=-1), dim=-1)

    def summary(self, print_params=False):
        summary(self, input_size=(self.in_features, 1, 1), print_params=print_params)

    @staticmethod
    def loss(pred, target):
        return NN.binary_cross_entropy(pred, target)


class Classifier(nn.Module):
    def __init__(self, nb_classes, in_features=1024,  leak=DEFAULT_LEAK):
        super().__init__()
        self.in_features = in_features
        self.nb_classes = nb_classes
        self.net = nn.Sequential(nn.Conv2d(in_features, 128, kernel_size=1, stride=1, padding=padding_same(1, 1)),
                                 nn.BatchNorm2d(128),
                                 nn.LeakyReLU(leak) if leak is not None else nn.ReLU(),
                                 nn.Conv2d(128, nb_classes, kernel_size=1, stride=1, padding=padding_same(1, 1)))

    def summary(self, print_params=False):
        summary(self, input_size=(self.in_features, 1, 1), print_params=print_params)

    def forward(self, x):
        return torch.mean(self.net(x), dim=[-2, -1], keepdim=False)
        # return torch.squeeze(torch.squeeze(self.net(x), dim=-1), dim=-1)

    @staticmethod
    def loss(pred, target, weight=None):
        if target.shape[-1] > 1:
            target = torch.argmax(target, dim=-1, keepdim=False).long()
        else:
            target = target[:, 0]
        ce = NN.cross_entropy(pred, target, reduce=None)
        if weight is not None:
            return (ce * weight).mean()
        else:
            return ce.mean()


class Evaluator(nn.Module):
    def __init__(self, in_features=1024, device=None, leak=DEFAULT_LEAK):
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.net = nn.Sequential(nn.Conv2d(in_features, 128, kernel_size=1, stride=1, padding=padding_same(1, 1)),
                                 nn.BatchNorm2d(128),
                                 nn.LeakyReLU(leak) if leak is not None else nn.ReLU(),
                                 nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=padding_same(1, 1)),
                                 nn.LeakyReLU(leak) if leak is not None else nn.ReLU())

    def forward(self, x, out_dis=None):
        return torch.mean(self.net(x), dim=[-2, -1], keepdim=False)
        # return torch.squeeze(torch.squeeze(self.net(x), dim=-1), dim=-1)

    def summary(self, print_params=False):
        summary(self, input_size=(self.in_features, 1, 1), print_params=print_params)

    @staticmethod
    def loss(pred, target, weight=None):
        mse = NN.mse_loss(pred, target, reduction='none')
        if weight is not None:
            mse *= weight
        return mse.sum()

#
# class CompleteDiscriminator(nn.Module):
#     def __init__(self, feat_extractor: nn.Module, device=None):
#         super().__init__()
#         self.device=device
#         self.feature_extractor = feat_extractor
#         self.discriminator = Discriminator(in_features=1024, device=device)
#
#
#     def forward(self, x):
#         feat = self.feature_extractor(x)
#         fake_confidence = self.discriminator(feat)
#         dist_type = self.classifier(feat)
#         dist_level = self.evaluator(feat)
#         return fake_confidence, dist_type, dist_level
#
# #
#
# class CompleteDiscriminator(nn.Module):
#     def __init__(self, nb_dist_categs, device=None):
#         # Feat-Extractor -> [Discriminator, Classifier, Evaluator]
#         super().__init__()
#         self.device=device
#         self.feature_extractor = FeatureExtractor(out_features=1024, device=device)
#         self.discriminator = Discriminator(in_features=1024, device=device)
#         self.classifier = Classifier(nb_dist_categs, in_features=1024, device=device)
#         self.evaluator = Evaluator(in_features=1024, device=device)
#
#     def forward(self, x):
#         feat = self.feature_extractor(x)
#         fake_confidence = self.discriminator(feat)
#         dist_type = self.classifier(feat)
#         dist_level = self.evaluator(feat)
#         return fake_confidence, dist_type, dist_level
