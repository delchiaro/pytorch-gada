import torch
from torch import nn
from numbers import Number
from fastorch import fastSequential
from torch.nn import functional as NN
def NP(tensor):
    return tensor.detach().cpu().numpy()

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def padding_same(filter_size, stride):
    return max(filter_size-stride, 0)//2


class PaddingSame2d(nn.Module):
    def __init__(self, next_kernel_size, next_stride, mode='replicate', const_value=0):
        """
        :param mode: 'constant', 'reflect' or 'replicate'
        :param const_value: fill value for 'constant' padding. Default: 0
        """
        super().__init__()
        self.mode = mode
        self.const_value = const_value
        next_stride = (next_stride, next_stride) if isinstance(next_stride, Number) else next_stride
        next_kernel_size = (next_kernel_size, next_kernel_size) if isinstance(next_kernel_size, Number) else next_kernel_size
        self.stride = next_stride
        self.filter_size = next_kernel_size

    def forward(self, img):
        in_height = img.shape[-2]
        in_width = img.shape[-1]
        pad_h = 0
        pad_w = 0

        #if self.filter_size[0] != in_height: # TODO: does tensorflow make this check?
        if in_height % self.stride[0] == 0:
            pad_h = max(self.filter_size[0] - self.stride[0], 0)
        else:
            pad_h = max(self.filter_size[0] - (in_height % self.stride[0]), 0)

        #if self.filter_size[1] != in_width: # TODO: does tensorflow make this check?
        if in_width % self.stride[1] == 0:
            pad_w = max(self.filter_size[1] - self.stride[1], 0)
        else:
            pad_w = max(self.filter_size[1] - (in_width % self.stride[1]), 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return NN.pad(img, pad=[pad_top, pad_bottom, pad_left, pad_right], mode=self.mode, value=self.const_value)

class PaddedSameConv2d(nn.modules.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True,
                 pad_mode='replicate', pad_const_value=0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        self.same_padding=PaddingSame2d(kernel_size, stride, mode=pad_mode, const_value=pad_const_value)

    def forward(self, img):
        img = self.same_padding(img)
        return super().forward(img)


def conv_out_size(input_size, filter_size, stride, padding):
    return (input_size - filter_size + 2 * padding) / stride + 1


def conv2d_out_shape(input_size, conv2d: nn.Conv2d):
    return [conv2d.out_channels,
            conv_out_size(input_size[0], conv2d.kernel_size[0], conv2d.stride[0], conv2d.padding[0]),
            conv_out_size(input_size[1], conv2d.kernel_size[1], conv2d.stride[1], conv2d.padding[1])]








class Encoder(nn.Module):
    def __init__(self, in_channels=3, device=None):
        super().__init__()
        self.device=device
        self.filters = [64, 128, 128]
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, self.filters[0], 4, 2, padding=padding_same(4, 2)),
                                   nn.LeakyReLU(.2))
        self.conv2 = nn.Sequential(nn.Conv2d(self.filters[0], self.filters[1], 4, 2, padding_same(4, 2)),
                                   nn.BatchNorm2d(self.filters[1]),
                                   nn.LeakyReLU(.2))
        self.conv3 = nn.Sequential(nn.Conv2d(self.filters[1], self.filters[2], 4, 2, padding_same(4, 2)),
                                   nn.BatchNorm2d(self.filters[2]),
                                   nn.LeakyReLU(.2))
    def out_channels(self):
        return self.filters[-1]

    def forward(self, x):
        conv1_emb = self.conv1(x)
        conv2_emb = self.conv2(conv1_emb)
        conv3_emb = self.conv3(conv2_emb)
        return  conv1_emb, conv2_emb, conv3_emb


class Decoder(nn.Module):
    def __init__(self, input_channels, hidden_skip_connection=True, device=None):
        super().__init__()
        self.device=device
        filters = [128, 64, 64, 3]
        m = 2 if hidden_skip_connection else 1
        self.deconv_3 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                      nn.ReflectionPad2d(padding=1),
                                      nn.Conv2d(input_channels, filters[0], kernel_size=3, stride=1, padding=0),
                                      nn.BatchNorm2d(filters[0]),
                                      nn.ReLU())

        self.deconv_2 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                      nn.ReflectionPad2d(padding=1),
                                      nn.Conv2d(filters[0]*m, filters[1], kernel_size=3, stride=1, padding=0),
                                      nn.BatchNorm2d(filters[1]),
                                      nn.ReLU())

        self.deconv_1 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                      nn.ReflectionPad2d(padding=1),
                                      nn.Conv2d(filters[1]*m, filters[2], kernel_size=3, stride=1, padding=0),
                                      nn.BatchNorm2d(filters[2]),
                                      nn.ReLU())

        # TODO: the out deconv has kernel_size=1?? 1by1 deconv??
        self.deconv_out = nn.Sequential(nn.Conv2d(filters[2], filters[3], kernel_size=1, stride=1, padding=0),
                                        nn.Sigmoid())

    def out_channels(self):
        return self.filters[-1]

    def forward(self, conv1_emb, conv2_emb, conv3_emb):
        out = self.deconv_3(conv3_emb)
        out = self.deconv_2(torch.cat((out, conv2_emb), dim=1))
        out = self.deconv_1(torch.cat((out, conv1_emb), dim=1))
        out = self.deconv_out(out)
        return out


class Generator(nn.Module):
    def __init__(self, nb_distortion, z_noise_dim, device=None):
        super().__init__()
        level_out_emb_size=32
        categ_out_emb_size=32
        self.z_dim = z_noise_dim
        self.dist_level_input_embedding, _ = fastSequential([nb_distortion, 'relu', nb_distortion, 'relu'], input_shape=1)
        self.dist_level_output_embedding, _ = fastSequential([32, 'relu', level_out_emb_size, 'relu'], input_shape=1)
        self.dist_categ_output_embedding, _ = fastSequential([32, 'relu', categ_out_emb_size, 'relu'], input_shape=nb_distortion)
        self.encoder = Encoder(in_channels=3+nb_distortion*2)
        self.decoder = Decoder(input_channels=self.encoder.out_channels()+z_noise_dim+level_out_emb_size+categ_out_emb_size)
        self.device = device

    def forward(self, img, dist_level, dist_categ):
        d_level_emb_in = self.dist_level_input_embedding(dist_level)
        d_categ_level = torch.cat((dist_categ, d_level_emb_in), dim=1)
        d_categ_level = d_categ_level.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, img.shape[2], img.shape[3])
        x = torch.cat((img, d_categ_level), dim=1)

        # Encoding the input image (concatenated with an embedding of distortion level/category)
        enc1, enc2, enc3 = self.encoder(x)

        # Embedding distortion level and category
        d_level_emb_out = self.dist_level_output_embedding(dist_level)
        d_categ_emb_out = self.dist_categ_output_embedding(dist_categ)
        d_categ_level_emb_out = torch.cat((d_categ_emb_out, d_level_emb_out), dim=1)
        d_categ_level_emb_out = d_categ_level_emb_out.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, enc3.shape[2], enc3.shape[3])

        # Preparing decoder input using the embedding of distortion level/category
        # concatenated to the encoder output and a noise vector.
        z_noise = torch.rand((enc3.shape[0], self.z_dim, enc3.shape[2], enc3.shape[3])).to(self.device)
        decoder_input = torch.cat((enc3, d_categ_level_emb_out, z_noise), dim=1)

        # Decoding
        decoded = self.decoder(enc1, enc2, decoder_input)
        return decoded

    @staticmethod
    def loss(generated_dist_img, real_dist_img):
        return torch.mean(torch.abs(real_dist_img-generated_dist_img))


class FeatureExtractor(nn.Module):
    def __init__(self, out_features=1024, device=None):
        super().__init__()
        self.device=device
        in_ch = 3
        self.net = nn.Sequential(PaddedSameConv2d(in_ch, 64, kernel_size=4, stride=2), nn.LeakyReLU(.2),
                                 PaddedSameConv2d(64, 128, kernel_size=4, stride=2), nn.LeakyReLU(.2),
                                 PaddedSameConv2d(128, 128, kernel_size=4, stride=2), nn.LeakyReLU(.2),
                                 nn.Conv2d(128, out_features, kernel_size=16, stride=1),
                                 nn.BatchNorm2d(out_features), nn.LeakyReLU(.2))


    def forward(self, *input):
        return self.net(*input)


import torch.nn.functional as NN
class Discriminator(nn.Module):
    def __init__(self, in_features=1024, device=None):
        super().__init__()
        self.device=device
        self.net = nn.Sequential(nn.Conv2d(in_features, 1, kernel_size=1, stride=1, padding=padding_same(1, 1)),
                                 nn.Sigmoid())
    def forward(self, x):
        #return torch.sigmoid(torch.mean(self.net(input), dim=[2,3], keepdim=False))
        return torch.squeeze(torch.squeeze(self.net(x), dim=-1), dim=-1)

    @staticmethod
    def loss(pred, target):
        return NN.binary_cross_entropy(pred, target)


class Classifier(nn.Module):
    def __init__(self, out_features, in_features=1024, device=None):
        super().__init__()
        self.device=device
        self.net = nn.Sequential(nn.Conv2d(in_features, 128, kernel_size=1, stride=1, padding=padding_same(1, 1)),
                                 nn.BatchNorm2d(128),
                                 nn.LeakyReLU(.2),
                                 nn.Conv2d(128, out_features, kernel_size=1, stride=1, padding=padding_same(1, 1)))

    def forward(self, x):
        #return torch.mean(self.net(input), dim=[-2, -1], keepdim=False)
        return torch.squeeze(torch.squeeze(self.net(x), dim=-1), dim=-1)

    @staticmethod
    def loss(pred, target):
        if target.shape[-1] > 1:
            target = torch.argmax(target, dim=-1, keepdim=False).long()
        return NN.cross_entropy(pred, target)


class Evaluator(nn.Module):
    def __init__(self, in_features=1024, device=None):
        super().__init__()
        self.device=device
        self.net = nn.Sequential(nn.Conv2d(in_features, 128, kernel_size=1, stride=1, padding=padding_same(1, 1)),
                                 nn.BatchNorm2d(128),
                                 nn.LeakyReLU(.2),
                                 nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=padding_same(1, 1)),
                                 #nn.ReLU() ## double-lrelu
                                 nn.LeakyReLU(.2)
                                 )

    def forward(self, x, out_dis=None):
        #self.net(torch.cat((x, out_dis), dim=1))
        #return torch.mean(self.net(x), dim=[-2, -1], keepdim=False)
        return torch.squeeze(torch.squeeze(self.net(x), dim=-1), dim=-1)

    @staticmethod
    def loss(pred, target):
        return NN.mse_loss(pred, target)


class CompleteDiscriminator(nn.Module):
    def __init__(self, nb_distortions, device=None):
        # Feat-Extractor -> [Discriminator, Classifier, Evaluator]
        super().__init__()
        self.device=device
        self.feature_extractor = FeatureExtractor(out_features=1024, device=device)
        self.discriminator = Discriminator(in_features=1024, device=device)
        self.classifier = Classifier(nb_distortions, in_features=1024, device=device)
        self.evaluator = Evaluator(in_features=1024, device=device)

    def forward(self, x):
        feat = self.feature_extractor(x)
        fake_confidence = self.discriminator(feat)
        dist_type = self.classifier(feat)
        dist_level = self.evaluator(feat)
        return fake_confidence, dist_type, dist_level

