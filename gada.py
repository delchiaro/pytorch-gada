from enum import Enum
from functools import reduce
from itertools import chain
from os.path import join
from typing import List
import torch
from tensorboardX import SummaryWriter
from torch import nn

from fastorch import fastSequential
from fastorch.summary import summary
from gada.networks import UNet, FullyConvFeatExtractor, FullyConvFC
from gada.utils import to_float32, to_device, index_to_1hot
from datasets.gadaset import GADAsetFactory
import time, datetime
from torch.utils.data import DataLoader


class TrainStepStyle(Enum):
    pytorch = 0
    tensorflow = 1


#crop_pred_size = None,  # (128,1) is the default --> discr_last_kernel = 16 with 3 convs.


# if discr_filters is None:
#     discr_filters = gen_filters
#
# if crop_pred_size is not None:
#     if discr_last_kernel != None:
#         raise ValueError("crop_pred_sizes is used to automatically compute discr_last_kernel, can't use both parameters.")
#     discr_last_kernel = (crop_pred_size[0] / (2 ** len(discr_filters))) / crop_pred_size[1]
#
#     print(f"taregt crop_size = {crop_pred_size[0]}")
#     print(f"taregt pred_size = {crop_pred_size[1]}")
#     print(f"gen_filters = {gen_filters}   --> nb_convs={len(discr_filters)} --> out_dim={crop_pred_size[0] / (2 ** len(discr_filters))}")
#     print(f"discr_last_kernel = {discr_last_kernel}")
#     if not discr_last_kernel.is_integer():
#         raise ValueError(f"crop_pred_sizes should compute an integer to be assigned to discr_last_kernel. Selected combination"
#                          f"of dimensions don't give an integer:\n"
#                          f"Please use a target crop_dim and target pred_dim that is an exponent of 2 and greater than: \n"
#                          f"   (2^nb_convs)*pred_dim   i.e.: 2^{len(discr_filters)} * {crop_pred_size[1]}\n")
#
#     discr_last_kernel = int(discr_last_kernel)
#
class GADA(nn.Module):
    G_LAMBDA_RECONSTRUCTION = 255


    def __init__(self,
                 nb_distortions=24,
                 z_noise_dim=96,
                 lr=1e-4,
                 target_im_size=128, # (minimum) width and height of input images
                 g_blocks=(64, 128, 256, 256),
                 f_blocks=None,
                 dist_level_emb_size=32,
                 dist_categ_emb_size=32,
                 discr_last_kernel=None,  # 16 is the default with 3 convs for discriminator, so that 128x128 crops become 16x16 -> 1x1 !
                 train_step_style=TrainStepStyle.pytorch,
                 g_disable_skip_connections=False,
                 device=None):
        super().__init__()
        self.device = device
        self.nb_distortions=nb_distortions
        self.target_im_size=target_im_size
        #
        #
        # self.F = FeatureExtractor(out_features=1024, filters=discr_filters, last_kernel_size=discr_last_kernel, device=device)
        # self.D = Discriminator(in_features=1024, device=device)
        # self.C = Classifier(nb_distortions, in_features=1024, device=device)
        # self.E = Evaluator(in_features=1024, device=device)
        # self.G = Generator(nb_distortions, z_noise_dim, img_channels=3, filters=gen_filters, device=device)
        f_blocks = g_blocks if f_blocks is None else f_blocks
        self.g_blocks = g_blocks
        self.f_blocks = f_blocks
        self.z_dim = z_noise_dim

        self.dist_level_in_embedding, self.dist_level_in_embedding_size = \
            fastSequential([nb_distortions, 'relu', nb_distortions, 'relu'], input_shape=1)
        self.dist_level_out_embedding, self.dist_level_out_embedding_size = \
            fastSequential([dist_level_emb_size, 'relu', dist_level_emb_size, 'relu'], input_shape=1)
        self.dist_categ_out_embedding, self.dist_categ_out_embedding_size = \
            fastSequential([dist_categ_emb_size, 'relu', dist_categ_emb_size, 'relu'], input_shape=nb_distortions)

        self.G = UNet(g_blocks,
                      input_channels=3+nb_distortions*2,
                      output_channels=3,
                      cat_embedding_channels=dist_level_emb_size + dist_categ_emb_size + z_noise_dim,
                      disable_skip_connections=g_disable_skip_connections,
                      append_layer=nn.Sigmoid())

        self.F = FullyConvFeatExtractor(target_im_size, f_blocks, input_channels=3, disable_full_size_conv=True)
        self.D = FullyConvFC(self.F.output_channels, 1, append_layer=nn.Sigmoid())
        self.C = FullyConvFC(self.F.output_channels, nb_distortions)
        self.E = FullyConvFC(self.F.output_channels, 1, append_layer=nn.Sigmoid())

        import torch.distributions as tdist
        self.z_noise_distribution = tdist.Normal(0, 0.1) # mean=0, stddev=0.1

        self.train_step_style = train_step_style

        self.opt_d = torch.optim.Adam(chain(self.F.parameters(), self.D.parameters(), self.E.parameters(), self.C.parameters()),
                                      lr=lr * .1, betas=(0.5, 0.999))
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=lr * 1, betas=(0.5, 0.999))
        if train_step_style is TrainStepStyle.tensorflow:
            self.opt_q = torch.optim.Adam(self.parameters(), lr=lr * 1, betas=(0.5, 0.999))
        elif train_step_style is TrainStepStyle.pytorch:
            self.opt_q = None

        self.to(device)


    def summary(self, crop_size=128, channels=3, print_params=False):
        #x = torch.zeros(2, channels+self.nb_dist_categs*2, crop_size, crop_size)
        x_ref = torch.zeros(2, channels, crop_size, crop_size, device=self.device)
        y_dist_categ = torch.zeros(2, 1, device=self.device)
        y_dist_level = torch.zeros(2, 1, device=self.device)

        im_shape = x_ref.shape

        enc_emb, dec_emb = self.distortion_embeddings(y_dist_categ, y_dist_level, im_shape)
        z_noise = self.generate_znoise_vector(im_shape)

        encoder_input = torch.cat((x_ref, enc_emb), dim=1)
        decoder_extra_input = torch.cat((dec_emb, z_noise), dim=1)

        print("\n\n\nGenerator: ")
        fake_img = summary(self.G, input=(encoder_input, decoder_extra_input), print_params=print_params)

        print("\n\n\nFeat-Extractor: ")
        fake_feat = summary(self.F, input=fake_img, print_params=print_params)

        print("\n\n\nDiscriminator: ")
        summary(self.D, input=fake_feat, print_params=print_params)

        print("\n\n\nClassifier: ")
        summary(self.C,  input=fake_feat, print_params=print_params)

        print("\n\n\nEvaluator: ")
        summary(self.E, input=fake_feat, print_params=print_params)



    def encoder_shape(self, im_tensor_shape):
        return self.G.compute_encoded_shape(im_tensor_shape)
        #return (im_tensor_shape[0], self.g_blocks[-1], im_tensor_shape[2] // 2 ** len(self.g_blocks), im_tensor_shape[3] // 2 ** len(self.g_blocks))

    def distortion_embeddings(self, y_dist_categ, y_dist_level, im_tensor_shape):
        ################## ENCODER Input Embedding #####################
        dcateg_onehot = index_to_1hot(y_dist_categ[:, 0].long(), self.nb_distortions).float()
        dlevel_input_emb = self.dist_level_in_embedding(y_dist_level)
        dcateg_dlevel_enc_emb = torch.cat((dcateg_onehot, dlevel_input_emb), dim=1)
        dcateg_dlevel_enc_emb = dcateg_dlevel_enc_emb.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, im_tensor_shape[2], im_tensor_shape[3]))

        ################## DECODER Input Embedding #####################
        enc_shape = self.encoder_shape(im_tensor_shape)
        # Embedding distortion level and category
        dlevel_emb_out = self.dist_level_out_embedding(y_dist_level)
        dcateg_emb_out = self.dist_categ_out_embedding(dcateg_onehot)
        dcateg_dlevel_dec_emb = torch.cat((dcateg_emb_out, dlevel_emb_out), dim=1)
        dcateg_dlevel_dec_emb = dcateg_dlevel_dec_emb.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, enc_shape[2], enc_shape[3]))
        # Preparing decoder input using the embedding of distortion level/category concatenated to the encoder output and a noise vector.

        return dcateg_dlevel_enc_emb, dcateg_dlevel_dec_emb

    def generate_znoise_vector(self, im_tensor_shape):
        enc_shape = self.encoder_shape(im_tensor_shape)
        return self.z_noise_distribution.sample((enc_shape[0], self.z_dim, enc_shape[2], enc_shape[3])).to(self.device)


    def generator_step(self, x_ref, y_dist_categ, y_dist_level, z_noise=None):
        im_shape = x_ref.shape

        z_noise = self.generate_znoise_vector(im_shape) if z_noise is None else z_noise
        enc_emb, dec_emb = self.distortion_embeddings(y_dist_categ, y_dist_level, im_shape)
        encoder_input = torch.cat((x_ref, enc_emb), dim=1)
        decoder_extra_input = torch.cat((dec_emb, z_noise), dim=1)

        fake_img = self.G.forward(encoder_input, decoder_extra_input)
        return fake_img


    def forward(self, x_ref, x_dist, y_dist_categ, y_dist_level):
        im_shape = x_ref.shape

        z_noise = self.generate_znoise_vector(im_shape)
        fake_img = self.generator_step(x_ref, y_dist_categ, y_dist_level, z_noise)

        # Feature Extraction
        fake_feat = self.F.forward(fake_img)
        real_feat = self.F.forward(x_dist)

        # Discriminator
        fake_confidence = self.D.forward(fake_feat)
        real_confidence = self.D.forward(real_feat)

        # Classifier
        fake_dist_type = self.C.forward(fake_feat)
        real_dist_type = self.C.forward(real_feat)

        # Evaluator
        fake_dist_lvl = self.E.forward(fake_feat)
        real_dist_lvl = self.E.forward(real_feat)

        fake_pred = [fake_img, fake_feat, fake_confidence, fake_dist_type, fake_dist_lvl]
        real_pred = [x_dist, real_feat, real_confidence, real_dist_type, real_dist_lvl]
        return fake_pred, real_pred


    def train_step(self, x_ref, x_dist, y_dist_categ, y_dist_level, ret_losses=True):
        if self.train_step_style is TrainStepStyle.pytorch:
            return self._train_step_pytorch(x_ref, x_dist, y_dist_categ, y_dist_level, ret_losses)
        elif self.train_step_style is TrainStepStyle.tensorflow:
            return self._train_step_tensorflow(x_ref, x_dist, y_dist_categ, y_dist_level, ret_losses)
        else:
            raise ValueError

    def _train_step_tensorflow(self, x_ref, x_dist, y_dist_categ, y_dist_level, ret_losses=True):
        im_shape = x_ref.shape

        z_noise = self.generate_znoise_vector(im_shape, im_shape[0])
        fake_img = self.generator_step(x_ref, y_dist_categ, y_dist_level, z_noise)

        zeros = torch.zeros(len(x_ref), 1)
        ones = torch.ones(len(x_ref), 1)

        adversarial_criterion = nn.BCELoss()
        classification_criterion = nn.CrossEntropyLoss()
        reconstruction_criterion =  lambda im_pred, im_true: (im_pred-im_true).abs().mean()
        evaluation_criterion = lambda y_pred, y_true: (y_pred-y_true).abs().mean()
        #evaluator_criterion = lambda y_pred, y_true:  NN.mse_loss(y_pred, y_true)

        ############################################## D Losses
        self.opt_d.zero_grad()
        # Discriminator on Fake features
        d_fake_feat = self.F.forward(fake_img.detach())  # stop backprop to the generator by detaching fake_img
        d_confidence_fake = self.D.forward(d_fake_feat)
        d_spot_g_loss = adversarial_criterion(d_confidence_fake, zeros)

        # Discriminator on Real features
        d_real_feat = self.F.forward(x_dist)
        d_confidence_real = self.D.forward(d_real_feat)
        d_spot_real_loss = adversarial_criterion(d_confidence_real, ones)

        d_loss = d_spot_g_loss + d_spot_real_loss
        d_loss.backward()
        self.opt_d.step()


        ############################################## G Loss
        self.opt_g.zero_grad()
        g_fake_feat = self.F.forward(fake_img)
        g_fake_confidence = self.D.forward(g_fake_feat)

        g_loss_l1 = reconstruction_criterion(fake_img, x_dist) * GADA.G_LAMBDA_RECONSTRUCTION
        g_loss_fooling_d = adversarial_criterion(g_fake_confidence, ones)
        g_loss = g_loss_l1 + g_loss_fooling_d

        g_loss.backward(retain_graph=True)
        self.opt_g.step()


        ############################################## C-E (Q) Losse
        self.opt_q.zero_grad()
        q_fake_feat = self.F.forward(fake_img)
        q_dist_categ_fake = self.C.forward(q_fake_feat)
        q_dist_level_fake = self.E.forward(q_fake_feat)
        qc_loss_fake =  classification_criterion(q_dist_categ_fake, y_dist_categ)
        qe_loss_fake =  evaluation_criterion(q_dist_level_fake, y_dist_level)

        q_real_feat = self.F.forward(x_dist)
        q_dist_categ_real = self.C.forward(q_real_feat)
        q_dist_level_real = self.E.forward(q_real_feat)
        qc_loss_real = classification_criterion(q_dist_categ_real, y_dist_categ)
        qe_loss_real = evaluation_criterion(q_dist_level_real, y_dist_level)

        q_loss = qc_loss_fake + qc_loss_real + qe_loss_fake + qe_loss_real
        q_loss.backward()
        self.opt_q.step()


        if ret_losses:
            losses = {'1_G/G_L1': g_loss_l1.detach().cpu(),
                      '1_G/G_fool_D': g_loss_fooling_d.detach().cpu(),
                      '1_G/G__loss': g_loss.detach().cpu(),

                      '2_D/D_spot_G': d_spot_g_loss.detach().cpu(),
                      '2_D/D_spot_real': d_spot_real_loss.detach().cpu(),
                      '2_D/D__loss': d_loss.detach().cpu(),

                      '3_Q/C_fake': qc_loss_fake.detach().cpu(),
                      '3_Q/C_real': qc_loss_real.detach().cpu(),
                      '3_Q/E_fake': qe_loss_fake.detach().cpu(),
                      '3_Q/E_real': qe_loss_real.detach().cpu(),
                      '3_Q/Q__loss': q_loss.detach().cpu()
                      }
        else:
            losses = None

        fake_pred = [fake_img, q_fake_feat, d_confidence_fake, q_dist_categ_fake, q_dist_level_fake]
        real_pred = [x_dist, q_real_feat, d_confidence_real, q_dist_categ_real, q_dist_level_real]
        return fake_pred, real_pred, losses


    def _train_step_pytorch(self, x_ref, x_dist, y_dist_categ, y_dist_level, ret_losses=True):
        im_shape = x_ref.shape

        wrong = torch.zeros(len(x_ref), 1, device=self.device)
        correct = torch.ones(len(x_ref), 1, device=self.device)
        # -----------------
        #  Train Generator
        # -----------------
        #set_requires_grad([self.D, self.F, self.C, self.E], False)
        #set_requires_grad(self.G, True)
        #fake_img = self.G(x_ref, y_dist_categ, y_dist_level)
        self.opt_g.zero_grad()
        z_noise = self.generate_znoise_vector(im_shape)
        fake_img = self.generator_step(x_ref, y_dist_categ, y_dist_level, z_noise)
        fake_feat = self.F.forward(fake_img)
        fake_confidence = self.D.forward(fake_feat)
        fake_dist_categ = self.C.forward(fake_feat)
        fake_dist_lvl = self.E.forward(fake_feat)

        adversarial_criterion = nn.BCELoss()
        classification_criterion = nn.CrossEntropyLoss()
        reconstruction_criterion =  lambda im_pred, im_true: (im_pred-im_true).abs().mean()
        evaluation_criterion = lambda y_pred, y_true: (y_pred-y_true).abs().mean()
        #evaluator_criterion = lambda y_pred, y_true:  NN.mse_loss(y_pred, y_true)


        loss_g_l1 = reconstruction_criterion(fake_img, x_dist) * GADA.G_LAMBDA_RECONSTRUCTION
        loss_g_e_fake = evaluation_criterion(fake_dist_lvl, y_dist_level)
        loss_g_fooling_d = adversarial_criterion(fake_confidence, correct)
        loss_g_c_fake = classification_criterion(fake_dist_categ, y_dist_categ[:,0])

        #loss_g = loss_g_fooling_d/2 + (loss_g_l1 + loss_g_c_fake + loss_g_e_fake)/2

        loss_g = 2*(loss_g_fooling_d/2) + 1*(loss_g_l1/4 + loss_g_c_fake/8 + loss_g_e_fake/8)
        loss_g.backward()
        self.opt_g.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------
        #set_requires_grad(self.G, False)
        #set_requires_grad([self.D, self.F, self.C, self.E], True)
        self.opt_d.zero_grad()

        # Loss for real images
        real_feat = self.F.forward(x_dist)
        real_confidence = self.D.forward(real_feat)
        real_dist_categ = self.C.forward(real_feat)
        real_dist_lvl = self.E.forward(real_feat)

        loss_d_detect_real = adversarial_criterion(real_confidence, correct)
        loss_c_real = classification_criterion(real_dist_categ, y_dist_categ[:,0])
        loss_e_real = evaluation_criterion(real_dist_lvl, y_dist_level)
        #loss_dce_real = loss_d_detect_real + loss_c_real + loss_e_real

        # Loss for fake images
        fake_feat = self.F.forward(fake_img.detach())  # stop backprop to the generator by detaching fake_img
        fake_confidence = self.D.forward(fake_feat)
        fake_dist_categ = self.C.forward(fake_feat)
        fake_dist_lvl = self.E.forward(fake_feat)

        loss_d_detect_g = adversarial_criterion(fake_confidence, wrong)
        loss_c_fake = classification_criterion(fake_dist_categ, y_dist_categ[:,0])
        loss_e_fake = evaluation_criterion(fake_dist_lvl, y_dist_level)
        #loss_dce_fake = loss_d_detect_g + loss_c_fake + loss_e_fake

        #loss_dce = loss_dce_real/2 + loss_dce_fake/2
        #loss_dce = loss_dce_real + loss_dce_fake


        loss_c = loss_c_real
        loss_e = loss_e_real
        #loss_c = (loss_c_real + loss_c_fake)/2
        #loss_e = (loss_e_real + loss_e_fake)/2
        loss_dce = loss_d_detect_g/4  + loss_d_detect_real/4 + loss_c/4 + loss_e/4
        loss_dce.backward()
        self.opt_d.step()



        if ret_losses:
            losses = {'1_G/G_L1': loss_g_l1.detach().cpu(),
                      '1_G/G_fool_D': loss_g_fooling_d.detach().cpu(),
                      '1_G/G__loss': loss_g.detach().cpu(),

                      '2_D/D_spot_G': loss_d_detect_g.detach().cpu(),
                      '2_D/D_spot_real': loss_d_detect_real.detach().cpu(),
                      '2_D/D__loss': (loss_d_detect_g+loss_d_detect_real).detach().cpu(),

                      '3_Q/C_fake': loss_c_fake.detach().cpu(),
                      '3_Q/C_real': loss_c_real.detach().cpu(),
                      '3_Q/E_fake': loss_e_fake.detach().cpu(),
                      '3_Q/E_real': loss_e_real.detach().cpu(),
                      '3_Q/Q__loss': loss_dce.detach().cpu()
                      }
        else:
            losses = None

        fake_pred = [fake_img, fake_feat, fake_confidence, fake_dist_categ, fake_dist_lvl]
        real_pred = [x_dist, real_feat, real_confidence, real_dist_categ, real_dist_lvl]
        return fake_pred, real_pred, losses


    def train_step_evaluator(self, x, y_dist_categ, y_dist_level, train_classifier=True):

        evaluation_criterion = lambda y_pred, y_true: (y_pred-y_true).abs().mean()
        classification_criterion = nn.CrossEntropyLoss()

        #set_requires_grad(self.G, False)
        #set_requires_grad([self.D, self.F, self.C, self.E], True)
        self.opt_d.zero_grad()

        feat = self.F.forward(x)
        dist_lvl = self.E.forward(feat)
        dist_categ = self.C.forward(feat)
        loss_e = evaluation_criterion(dist_lvl, y_dist_level)
        loss_c = torch.Tensor([0.]).to(self.device)
        if train_classifier:
            loss_c = classification_criterion(dist_categ, y_dist_categ[:, 0])
        loss = loss_e*4 + loss_c
        loss.backward()
        self.opt_d.step()
        return loss_e.detach().cpu(), loss_c.detach().cpu()


import numpy as np


#%%
# def sum_loss_dict(loss_dict_a: dict, loss_dict_b: dict):
#     return {k: loss_dict_a[k]+loss_dict_b[k] for k in loss_dict_a.keys()}

def mean_all_losses(loss_dict_list: List[dict]):
    N = len(loss_dict_list)
    d = reduce(lambda loss_dict_a, loss_dict_b: {k: loss_dict_a[k]+loss_dict_b[k] for k in loss_dict_a.keys()}, loss_dict_list)
    return {k: v/N for k, v in d.items()}


def write_dict_summary(d: dict, writer: SummaryWriter, step):
    if writer is not None and d is not None:
        for k, v in d.items():
            writer.add_scalar(k, v, step)

def finetune_GADA(gada: GADA,
                  trainset: GADAsetFactory,
                  testset: GADAsetFactory,
                  gen_dist_from_trainset: bool,
                  use_dist_from_trainset: bool,
                  nb_epochs=1000,
                  bs=64,
                  crop_dim=128,
                  test_bs=None,
                  test_period=50,
                  data_in_gpu=False,
                  log_dir='./logs/eval_finetune/',
                  state_dir='./weights_finetune/',
                  first_save_state_epoch=0,
                  save_state_interval=50,
                  jpeg_grid=True,
                 train_classifier=True,
                  ):
    if not gen_dist_from_trainset and not use_dist_from_trainset:
        raise ValueError("At least oen of the 'gen_dist_from_trainset' and 'use_dist_from_trainset' must be True.")
    test_bs = bs if test_bs is None else test_bs
    num_workers = 3
    if data_in_gpu:
        train_dataset = trainset.tensor_dataset(crop_dim, device=gada.device, jpeg_grid=jpeg_grid)
        trainset.drop_images()
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=True)
        #
        test_dataset = testset.tensor_dataset(crop_dim, device=gada.device, jpeg_grid=jpeg_grid)
        testset.drop_images()
        test_loader =  DataLoader(test_dataset, batch_size=test_bs, shuffle=False, num_workers=num_workers, pin_memory=False)

    else:
        train_dataset = trainset.tensor_dataset(crop_dim, device=None, jpeg_grid=jpeg_grid)
        trainset.drop_images()
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

        test_dataset = testset.tensor_dataset(crop_dim, device=None, jpeg_grid=jpeg_grid)
        testset.drop_images()
        test_loader =  DataLoader(test_dataset, batch_size=test_bs, shuffle=False, num_workers=num_workers, pin_memory=True)


    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    epoch_summary_writer = None
    if log_dir is not None:
        epoch_summary_writer = SummaryWriter(join(log_dir, timestamp))

    if first_save_state_epoch == 0 and state_dir is not None:
        from os import makedirs
        state_dir = join(state_dir, timestamp)
        makedirs(state_dir, exist_ok=True)
        torch.save(gada.state_dict(), join(state_dir, f'gada-eval-finetune-[ep=0]'))


    for epoch in range(1, nb_epochs+1):
        eval_losses_dist = []
        class_losses_dist = []
        eval_losses_gen = []
        class_losses_gen = []
        epoch_start_time = time.time()
        gada.train()
        for batch, (x_ref, x_dist, y_dist_categ, y_dist_level, id) in enumerate(train_loader):
            if data_in_gpu:
                x_ref, x_dist, y_dist_level = to_float32(x_ref, x_dist, y_dist_level)
            else:
                x_ref, x_dist, y_dist_level = to_device(*to_float32(x_ref, x_dist, y_dist_level), device=gada.device)
                y_dist_categ = y_dist_categ.to(gada.device)

            if gen_dist_from_trainset:
                rand_dist_level = torch.empty_like(y_dist_level).uniform_(0, 1)
                rand_dist_categ = torch.randint(0, gada.nb_distortions-1, y_dist_categ.shape).to(gada.device)
                x_gen = gada.generator_step(x_ref, rand_dist_categ, rand_dist_level).to(gada.device)
                eval_loss_gen, class_loss_gen = gada.train_step_evaluator(x_gen, rand_dist_categ, rand_dist_level, train_classifier)
                eval_losses_gen.append(eval_loss_gen.numpy())
                class_losses_gen.append(class_loss_gen.numpy())
            if use_dist_from_trainset:
                eval_loss_dist, class_loss_dist = gada.train_step_evaluator(x_dist, y_dist_categ, y_dist_level, train_classifier)
                eval_losses_dist.append(eval_loss_dist.numpy())
                class_losses_dist.append(class_loss_dist.numpy())


        print(f"Epoch[{epoch}/{nb_epochs}] -- Execution Time:  {time.time()-epoch_start_time:.5}s:)\n")
        if gen_dist_from_trainset:
            eval_loss = np.mean(eval_losses_gen)
            class_loss = np.mean(class_losses_gen)
            if epoch_summary_writer is not None:
                epoch_summary_writer.add_scalar('E/loss_gen', eval_loss, epoch)
                epoch_summary_writer.add_scalar('C/loss_gen', class_loss, epoch)
            print(f" Gen img  -->  Eval Loss: {eval_loss:.5f}   -   Class Loss: {class_loss:.5f}")
        if use_dist_from_trainset:
            eval_loss = np.mean(eval_losses_dist)
            class_loss = np.mean(class_losses_dist)
            if epoch_summary_writer is not None:
                epoch_summary_writer.add_scalar('E/loss_discr', eval_loss, epoch)
                epoch_summary_writer.add_scalar('C/loss_discr', class_loss, epoch)
            print(f" Dist img -->  Eval Loss: {eval_loss:.5f}   -   Class Loss: {class_loss:.5f}")
        print("")

        if epoch%test_period == 0:
            print("TESTING ...")
            lcc, srocc = test_GADA(gada, test_loader=test_loader, bs=test_bs, crop_dim=crop_dim, data_in_gpu=data_in_gpu, verbose=0)
            if epoch_summary_writer is not None:
                epoch_summary_writer.add_scalar('test/lcc', lcc, epoch)
                epoch_summary_writer.add_scalar('test/srocc', srocc, epoch)
            print(f"lcc   = {lcc:.5f}")
            print(f"srocc = {srocc:.5f}")


        if  state_dir is not None and epoch > first_save_state_epoch and (epoch-first_save_state_epoch) % save_state_interval == 0:
            torch.save(gada.state_dict(), join(state_dir, f'gada-[ep={epoch}]'))





def train_GADA(gada: GADA,
               trainset: GADAsetFactory,
               testset: GADAsetFactory,
               #opt_fn: Callable[[Iterator], torch.optim.Optimizer],
               nb_epochs=1000,
               bs=64,
               crop_dim=128,
               test_bs=None,
               test_period=50,
               data_in_gpu=False,
               log_dir='./logs/',
               state_dir='./weights/',
               first_save_state_epoch=0,
               save_state_interval=50,
               jpeg_grid=True):

    test_bs = bs if test_bs is None else test_bs
    num_workers = 3
    if data_in_gpu:
        train_dataset = trainset.tensor_dataset(crop_dim, device=gada.device, jpeg_grid=jpeg_grid)
        trainset.drop_images()
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=True)
        #
        test_dataset = testset.tensor_dataset(crop_dim, device=gada.device, jpeg_grid=jpeg_grid)
        testset.drop_images()
        test_loader =  DataLoader(test_dataset, batch_size=test_bs, shuffle=False, num_workers=num_workers, pin_memory=False)

    else:
        train_dataset = trainset.tensor_dataset(crop_dim, device=None, jpeg_grid=jpeg_grid)
        trainset.drop_images()
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

        test_dataset = testset.tensor_dataset(crop_dim, device=None, jpeg_grid=jpeg_grid)
        testset.drop_images()
        test_loader =  DataLoader(test_dataset, batch_size=test_bs, shuffle=False, num_workers=num_workers, pin_memory=True)


    #opt = torch.optim.Adam(gada.parameters(), lr=0.0001, betas=(0.5, 0.999)) if opt_fn is None else opt_fn(gada.parameters())

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    if log_dir is not None:
        #batch_summary_writer = SummaryWriter(join(log_dir, timestamp, 'batch'))
        batch_summary_writer=None
        epoch_summary_writer = SummaryWriter(join(log_dir, timestamp))
    else:
        batch_summary_writer = epoch_summary_writer = None

    if first_save_state_epoch == 0 and state_dir is not None:
        from os import makedirs
        state_dir = join(state_dir, timestamp)
        makedirs(state_dir, exist_ok=True)
        torch.save(gada.state_dict(), join(state_dir, f'gada-[ep=0]'))


    for epoch in range(1, nb_epochs+1):
        nb_batches = len(train_loader)
        epoch_losses = []
        epoch_start_time = time.time()
        gada.train()
        for batch, (x_ref, x_dist, y_dist_categ, y_dist_level, id) in enumerate(train_loader):
            if data_in_gpu:
                x_ref, x_dist, y_dist_level = to_float32(x_ref, x_dist, y_dist_level)
            else:
                x_ref, x_dist, y_dist_level = to_device(*to_float32(x_ref, x_dist, y_dist_level), device=gada.device)
                y_dist_categ = y_dist_categ.to(gada.device)

            #x_ref /= 255.
            #x_dist /= 255.

            #fake_pred, real_pred, losses = gada.train_step_bongini(x_ref, x_dist, y_dist_categ, y_dist_level, ret_losses=True)
            fake_pred, real_pred, losses = gada.train_step(x_ref, x_dist, y_dist_categ, y_dist_level, ret_losses=True)
            #write_dict_summary(losses, batch_summary_writer, (epoch-1) * nb_batches + batch + 1)
            epoch_losses.append(losses)


        ep_loss = mean_all_losses(epoch_losses)
        write_dict_summary(ep_loss, epoch_summary_writer, epoch)
        print(f"Epoch[{epoch}/{nb_epochs}] -- Execution Time:  {time.time()-epoch_start_time:.5}s")
       # print(f"Mean losses:  G-l1={ep_loss['1_G/loss_reconstruction']:.4}   G-fool-D={ep_loss['1_G/loss_G_fool_D']:.4}   D-spot-G={ep_loss['2_D/loss_D_spot_G']:.4}")
       # print(f"              sC={ep_loss['3_C/loss']:.4}   E={ep_loss['4_E/loss']:.4}")
        print("")

        if epoch%test_period == 0:
            lcc, srocc = test_GADA(gada, test_loader=test_loader, bs=test_bs, crop_dim=crop_dim, data_in_gpu=data_in_gpu, verbose=0)
            if epoch_summary_writer is not None:
                epoch_summary_writer.add_scalar('test/lcc', lcc, epoch)
                epoch_summary_writer.add_scalar('test/srocc', srocc, epoch)


        if  state_dir is not None and epoch > first_save_state_epoch and (epoch-first_save_state_epoch) % save_state_interval == 0:
            torch.save(gada.state_dict(), join(state_dir, f'gada-[ep={epoch}]'))
    #
    # @property
    # def device():
    #     return list(gada.parameters())[0].device


def NP(tensor):
    return np.squeeze(tensor.detach().cpu().numpy())

def IMG(tensor):
    return NP(tensor).swapaxes(-3, -2).swapaxes(-2, -1)





def test_GADA(gada: GADA,
              testset: GADAsetFactory = None,
              test_loader=None,
              bs=64,
              crop_dim=128,
              data_in_gpu=False,
              state_path=None,
              verbose=0,
              nb_crops=30,
              jpeg_grid=True):

    if state_path is not None:
        gada.load_state_dict(torch.load(state_path, map_location=gada.device))

    if test_loader is None:
        if data_in_gpu:
            test_dataset = testset.tensor_dataset(crop_dim, device=gada.device, jpeg_grid=jpeg_grid)
            testset.drop_images()
            test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=2, pin_memory=False)

        else:
            test_dataset = testset.tensor_dataset(crop_dim, device=None, jpeg_grid=jpeg_grid)
            testset.drop_images()
            test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True if gada.device != "cpu" else False)

    gada.eval()

    lvl_preds_by_id = {}
    dist_lvl_by_id = {}
    dist_type_by_id = {}
    for i in range(nb_crops): # 30 epochs because we want to compute 30 random-crop prediction per each test image
        if verbose:
            print(f"Testing crops {i+1}")
        for batch, (x_ref, x_dist, y_dist_categ, y_dist_level, idx) in enumerate(test_loader):
            if data_in_gpu:
                x_ref, x_dist, y_dist_level = to_float32(x_ref, x_dist, y_dist_level)
            else:
                x_ref, x_dist, y_dist_level = to_device(*to_float32(x_ref, x_dist, y_dist_level), device=gada.device)
                y_dist_categ = y_dist_categ.to(gada.device)

            x_feats = gada.F(x_dist)
            dist_level_pred = gada.E(x_feats)
            #dist_type_pred  , dist_level_pred =

            #(fake_img_preds, real_img_preds), loss = gada.forward_w_loss(x_ref, x_dist, y_dist_type, y_dist_level)
            #_, confidence_pred, dist_type_pred, dist_level_pred = real_img_preds
            dist_level_acc = (dist_level_pred-y_dist_level).abs()

            dist_level_pred = NP(dist_level_pred)
            y_dist_level = NP(y_dist_level)
            y_dist_type = NP(y_dist_categ)
            for i, id in enumerate(idx):
                id = int(id)
                if id not in dist_lvl_by_id.keys():
                    lvl_preds_by_id[id] = []
                    dist_lvl_by_id[id] = []
                    dist_type_by_id[id] = []
                lvl_preds_by_id[id].append(dist_level_pred[i])
                dist_lvl_by_id[id].append(y_dist_level[i])
                dist_type_by_id[id].append(y_dist_type[i])

    from scipy.stats import pearsonr, spearmanr
    lvl_preds_by_id_means = {}
    for k in lvl_preds_by_id.keys():
        lvl_preds_by_id_means[k] = np.mean(lvl_preds_by_id[k])

    lvl_preds_by_id_means = np.array([lvl_preds_by_id_means[k] for k in sorted(lvl_preds_by_id_means.keys())])

    dist_lvl_by_id = {k: dist_lvl_by_id[k][0] for k in dist_lvl_by_id.keys()}

    dist_type_by_id = {k: dist_type_by_id[k][0] for k in dist_type_by_id.keys()}


    dist_lvl_by_id = np.array([dist_lvl_by_id[k] for k in sorted(dist_lvl_by_id.keys())])
    lcc_global = pearsonr(lvl_preds_by_id_means, dist_lvl_by_id)[0]
    srocc_global = spearmanr(lvl_preds_by_id_means, dist_lvl_by_id)[0]

    if verbose:
        print(f"LCC:  {lcc_global}")
        print(f"SROCC: {srocc_global}")

    return lcc_global, srocc_global



def plot_GADA_gen_imgs(gada: GADA,
              testset: GADAsetFactory = None,
              test_loader=None,
              bs=64,
              crop_dim=128,
              data_in_gpu=False,
              state_path=None,
                       timestamp=None,
              verbose=0,
              imgs_per_plot=4,
                    skip_batches=0,
                       filter_dist=tuple(),
              jpeg_grid=True):

    if state_path is not None:
        if timestamp is not None:
            state_path = join(state_path, timestamp)
        gada.load_state_dict(torch.load(state_path, map_location=gada.device))

    if test_loader is None:
        if data_in_gpu:
            test_dataset = testset.tensor_dataset(crop_dim, device=gada.device, jpeg_grid=jpeg_grid)
            testset.drop_images()
            test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)

        else:
            test_dataset = testset.tensor_dataset(crop_dim, device=None, jpeg_grid=jpeg_grid)
            testset.drop_images()
            test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True if gada.device != "cpu" else False)

    gada.eval()
    if skip_batches > 0:
        print(f"Skipping first {skip_batches} batches..")
    for batch, (x_ref, x_dist, y_dist_categ, y_dist_level, idx) in enumerate(test_loader):
        if skip_batches>0:
            skip_batches-=1
            continue
        if data_in_gpu:
            x_ref, x_dist, y_dist_level = to_float32(x_ref, x_dist, y_dist_level)
        else:
            x_ref, x_dist, y_dist_level = to_device(*to_float32(x_ref, x_dist, y_dist_level), device=gada.device)
            y_dist_categ = y_dist_categ.to(gada.device)

        #x_ref /= 255.
        #x_dist /= 255.

        generated = gada.generator_step(x_ref, y_dist_categ, y_dist_level).detach().cpu()
        #rec_loss = Generator.loss(generated, x_dist.detach().cpu()).detach().cpu()
        #rec_loss_b = Generator.loss(generated, x_ref.detach().cpu()).detach().cpu()
        #print(f"Reconstruction loss dist = {rec_loss*255}")
        #print(f"Reconstruction loss ref  = {rec_loss_b*255}")

        generated = IMG(generated)
        origs = IMG(x_dist)
        refs = IMG(x_ref)

        from matplotlib import pyplot as plt
        remaining = len(generated)

        offset=0
        while(remaining > 0):
            N = imgs_per_plot
            if remaining < imgs_per_plot:
                N = remaining
            fig, ax = plt.subplots(N, 3, figsize=(10, N*2))
            for i in range(N):
                ax[i,0].imshow(refs[i+offset])
                ax[i,1].imshow(origs[i+offset])
                ax[i,2].imshow(generated[i+offset])
            fig.show()
            remaining -= N
            offset+=N
    return
