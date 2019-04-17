from copy import deepcopy
from functools import reduce
from os.path import join
from typing import List, Callable

import torch
from dataclasses import dataclass, field

from tensorboardX import SummaryWriter
from torch import nn

from gada import Generator, CompleteDiscriminator, Discriminator, Classifier, Evaluator
from .gadaset import GADAsetFactory
import time, datetime


from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset

@dataclass
class LossStats:
    g_reconstruction_loss: torch.Tensor=torch.Tensor([0])
    g_fooling_d: torch.Tensor=torch.Tensor([0])

    d_loss_fake: torch.Tensor=torch.Tensor([0])
    d_loss_real: torch.Tensor=torch.Tensor([0])

    ac_loss_fake: torch.Tensor=torch.Tensor([0])
    ac_loss_real: torch.Tensor=torch.Tensor([0])

    ev_loss_fake: torch.Tensor=torch.Tensor([0])
    ev_loss_real: torch.Tensor=torch.Tensor([0])

    g_loss: torch.Tensor = field(init=False, repr=False, compare=False)
    ac_loss: torch.Tensor = field(init=False, repr=False, compare=False)
    d_loss: torch.Tensor = field(init=False, repr=False, compare=False)
    ev_loss: torch.Tensor = field(init=False, repr=False, compare=False)
    loss: torch.Tensor = field(init=False, repr=False, compare=False)


    def update(s):
        s.g_loss = GADA.G_LAMBDA_RECONSTRUCTION * s.g_reconstruction_loss + s.g_fooling_d
        s.ac_loss = s.ac_loss_fake + s.ac_loss_real
        s.d_loss = s.d_loss_fake + s.d_loss_real
        s.ev_loss = s.ev_loss_fake + s.ev_loss_real
        s.loss = s.g_loss + s.ac_loss + s.d_loss + s.ev_loss

    def summary_write(s, loss_writer: SummaryWriter, step, tag_1_prepend='', tag_1_append='', tag_2_prepend='', tag_2_append=''):
        t1p = tag_1_prepend
        t1a = tag_1_append
        t2p = tag_2_prepend
        t2a = tag_2_append

        loss_writer.add_scalar(f'{t1p}adversarial{t1a}/{t2p}G_fool_D{t2a}', s.g_fooling_d, step)
        loss_writer.add_scalar(f'{t1p}adversarial{t1a}/{t2p}D_sgama_G{t2a}', s.g_fooling_d, step)

        loss_writer.add_scalar(f'{t1p}G{t1a}/{t2p}loss{t2a}', s.g_loss, step)
        loss_writer.add_scalar(f'{t1p}G{t1a}/{t2p}reconstruction{t2a}', s.g_reconstruction_loss, step)
        loss_writer.add_scalar(f'{t1p}G{t1a}/{t2p}fooling{t2a}', s.g_fooling_d, step)

        loss_writer.add_scalar(f'{t1p}D{t1a}/{t2p}loss_fake{t2a}', s.d_loss_fake, step)
        loss_writer.add_scalar(f'{t1p}D{t1a}/{t2p}loss_real{t2a}', s.d_loss_real, step)
        loss_writer.add_scalar(f'{t1p}D{t1a}/{t2p}loss{t2a}', s.d_loss, step)

        loss_writer.add_scalar(f'{t1p}AC{t1a}/{t2p}loss_fake{t2a}', s.ac_loss_fake, step)
        loss_writer.add_scalar(f'{t1p}AC{t1a}/{t2p}loss_real{t2a}', s.ac_loss_real, step)
        loss_writer.add_scalar(f'{t1p}AC{t1a}/{t2p}loss{t2a}', s.ac_loss, step)

        loss_writer.add_scalar(f'{t1p}EV{t1a}/{t2p}loss_fake{t2a}', s.ev_loss_fake, step)
        loss_writer.add_scalar(f'{t1p}EV{t1a}/{t2p}loss_real{t2a}', s.ev_loss_real, step)
        loss_writer.add_scalar(f'{t1p}EV{t1a}/{t2p}loss{t2a}', s.ev_loss, step)

    def __post_init__(self):
        self.update()

    @classmethod
    def init_fields(cls):
        return [k for k, v in cls.__dataclass_fields__.items() if v.init]

    @staticmethod
    def sum_all(loss_stats: List) -> 'LossStats':
        init_fields = LossStats.init_fields()
        d = {k: reduce((lambda a, b: a.__dict__[k] + b.__dict__[k]), loss_stats) for k in init_fields}
        # {k: reduce((lambda a, b: a.__dict__[k]+b.__dict__[k]), loss_stats) for k in keys}
        return LossStats(**d)




class GADA(nn.Module):
    G_LAMBDA_RECONSTRUCTION = 150

    def __init__(self, nb_distortions=24, z_noise_dim=96, loss_writer: SummaryWriter=None, device=None):
        super().__init__()
        self.device = device
        self.nb_distortions=nb_distortions
        self.G = Generator(nb_distortions, z_noise_dim, device=device)
        self.D = CompleteDiscriminator(nb_distortions, device=device)
        self.loss_writer = loss_writer
        self.to(device)


    def forward(self, x_ref, x_dist, y_dist_type, y_dist_level):
        fake_img = self.G(x_ref, y_dist_level, y_dist_type)
        fake_img_confidence, fake_img_dist_type, fake_img_dist_level = self.D(fake_img)
        dist_img_confidence, dist_img_dist_type, dist_img_dist_level = self.D(x_dist)

        return [fake_img, fake_img_confidence, fake_img_dist_type, fake_img_dist_level],\
               [x_dist, dist_img_confidence, dist_img_dist_type, dist_img_dist_level]




    class Loss:
        def __init__(self, fake_img_preds, real_img_preds, y_dist_type, y_dist_level):
            fake_img, fake_img_confidence, fake_img_dist_type, fake_img_dist_level = fake_img_preds
            real_img, real_img_confidence, real_img_dist_type, real_img_dist_level = real_img_preds

            self.g_reconstruction_loss = Generator.loss(fake_img, real_img)
            self.g_fooling_d = Discriminator.loss(fake_img_confidence, torch.zeros_like(fake_img_confidence))  # g fooling d
            self.g_loss = GADA.G_LAMBDA_RECONSTRUCTION * self.g_reconstruction_loss + self.g_fooling_d

            self.d_loss_fake = Discriminator.loss(fake_img_confidence, torch.ones_like(fake_img_confidence))  # d detecting fake imgs
            self.d_loss_real = Discriminator.loss(real_img_confidence, torch.zeros_like(real_img_confidence))  # d detecting good imgs
            self.d_loss = self.d_loss_fake + self.d_loss_real

            self.ac_loss_fake = Classifier.loss(fake_img_dist_type, y_dist_type)
            self.ac_loss_real = Classifier.loss(real_img_dist_type, y_dist_type)
            self.ac_loss = self.ac_loss_fake + self.ac_loss_real

            self.ev_loss_fake = Evaluator.loss(fake_img_dist_level, y_dist_level)
            self.ev_loss_real = Evaluator.loss(real_img_dist_level, y_dist_level)
            self.ev_loss = self.ev_loss_fake + self.ev_loss_real

            self.loss = self.g_loss + self.ac_loss + self.d_loss + self.ev_loss


        @staticmethod
        def sum_all(loss_stats: List) -> 'GADA.Loss':
            res = deepcopy(loss_stats[0])
            fields = [k for k in res.__dict__.keys() if (not k.startswith('__') and not callable(res.__dict__[k]))]
            d = {k: reduce((lambda a, b: a.__dict__[k] + b.__dict__[k]), loss_stats) for k in fields}
            # {k: reduce((lambda a, b: a.__dict__[k]+b.__dict__[k]), loss_stats) for k in keys}
            for k, v in d.values():
                setattr(res, k, v)
            return res

        def summary_write(s, loss_writer: SummaryWriter, step, tag_1_prepend='', tag_1_append='', tag_2_prepend='', tag_2_append=''):
            t1p = tag_1_prepend
            t1a = tag_1_append
            t2p = tag_2_prepend
            t2a = tag_2_append

            loss_writer.add_scalar(f'{t1p}adversarial{t1a}/{t2p}G_fool_D{t2a}', s.g_fooling_d, step)
            loss_writer.add_scalar(f'{t1p}adversarial{t1a}/{t2p}D_sgama_G{t2a}', s.d_loss_fake, step)

            loss_writer.add_scalar(f'{t1p}G{t1a}/{t2p}loss{t2a}', s.g_loss, step)
            loss_writer.add_scalar(f'{t1p}G{t1a}/{t2p}loss_reconstruction{t2a}', s.g_reconstruction_loss, step)
            loss_writer.add_scalar(f'{t1p}G{t1a}/{t2p}loss_fooling{t2a}', s.g_fooling_d, step)

            loss_writer.add_scalar(f'{t1p}D{t1a}/{t2p}loss_fake{t2a}', s.d_loss_fake, step)
            loss_writer.add_scalar(f'{t1p}D{t1a}/{t2p}loss_real{t2a}', s.d_loss_real, step)
            loss_writer.add_scalar(f'{t1p}D{t1a}/{t2p}loss{t2a}', s.d_loss, step)

            loss_writer.add_scalar(f'{t1p}AC{t1a}/{t2p}loss_fake{t2a}', s.ac_loss_fake, step)
            loss_writer.add_scalar(f'{t1p}AC{t1a}/{t2p}loss_real{t2a}', s.ac_loss_real, step)
            loss_writer.add_scalar(f'{t1p}AC{t1a}/{t2p}loss{t2a}', s.ac_loss, step)

            loss_writer.add_scalar(f'{t1p}EV{t1a}/{t2p}loss_fake{t2a}', s.ev_loss_fake, step)
            loss_writer.add_scalar(f'{t1p}EV{t1a}/{t2p}loss_real{t2a}', s.ev_loss_real, step)
            loss_writer.add_scalar(f'{t1p}EV{t1a}/{t2p}loss{t2a}', s.ev_loss, step)


    def forward_w_loss(self, x_ref, x_dist, y_dist_type, y_dist_level):
        fake_img_preds, real_img_preds = self.forward(x_ref, x_dist, y_dist_type, y_dist_level)
        losses = GADA.Loss(fake_img_preds, real_img_preds, y_dist_type, y_dist_level)
        return (fake_img_preds, real_img_preds), losses

    def forward_w_loss2(self, x_ref, x_dist, y_dist_type, y_dist_level):
        fake_img_preds, real_img_preds = self.forward(x_ref, x_dist, y_dist_type, y_dist_level)
        fake_img, fake_img_confidence, fake_img_dist_type, fake_img_dist_level = fake_img_preds
        real_img, real_img_confidence, real_img_dist_type, real_img_dist_level = real_img_preds
        g_reconstruction_loss = Generator.loss(fake_img, real_img)
        g_fooling_d = Discriminator.loss(fake_img_confidence, torch.zeros_like(fake_img_confidence))  # g fooling d
        g_loss = GADA.G_LAMBDA_RECONSTRUCTION * g_reconstruction_loss + g_fooling_d

        d_loss_fake = Discriminator.loss(fake_img_confidence, torch.ones_like(fake_img_confidence))  # d detecting fake imgs
        d_loss_real = Discriminator.loss(real_img_confidence, torch.zeros_like(real_img_confidence))  # d detecting good imgs
        d_loss = d_loss_fake + d_loss_real

        ac_loss_fake = Classifier.loss(fake_img_dist_type, y_dist_type)
        ac_loss_real = Classifier.loss(real_img_dist_type, y_dist_type)
        ac_loss = ac_loss_fake + ac_loss_real

        ev_loss_fake = Evaluator.loss(fake_img_dist_level, y_dist_level)
        ev_loss_real = Evaluator.loss(real_img_dist_level, y_dist_level)
        ev_loss = ev_loss_fake + ev_loss_real

        return (fake_img_preds, real_img_preds), g_loss + ac_loss + d_loss + ev_loss

    def train_on_batch(self, x_ref, x_dist, y_dist_type, y_dist_level, optim: torch.optim.Optimizer):
        optim.zero_grad()
        _, losses = self.forward_w_loss(x_ref, x_dist, y_dist_type, y_dist_level)
        losses.loss.backward()
        #_, losses = self.forward_w_loss2(x_ref, x_dist, y_dist_type, y_dist_level)
        #losses.backward()
        optim.step()
        return losses


import numpy as np


#%%


def to_float32(*tensors):
    return tuple(t.float() for t in tensors)

def to_device(*tensors, device):
    return tuple(t.to(device) for t in tensors)

def index_to_1hot(indices, nb_elements):
    return torch.diag(torch.ones([nb_elements]))[indices]


def random_crop(img_batch, w=64, h=64):
    x = torch.randint(0, img_batch.shape[1], img_batch.shape[0])
    y = torch.randint(0, img_batch.shape[2], img_batch.shape[0])
    cropped = img_batch[:, x, y, :]

    torch.random(img_batch[0])
    point = torch.random(img_batch.shape[1]-w, img_batch.shape[2]-h)



from torch.nn import functional as NN
def random_crop(x, target_size=64):
    import matplotlib.pyplot as plt
    def show(x):
        x = x.cpu().detach().numpy().swapaxes(0, 1).swapaxes(1, 2)
        print(x.shape)
        plt.imshow(x)
        plt.show()

    def get_img(x, ch):
        x = np.concatenate([x[:,:,ch:ch+1], x[:,:,ch:ch+1], x[:,:,ch:ch+1]], axis=-1)
        x.swapaxes(0,1).swapaxes(1,2)
        return x

    def build_grid(source_w, source_h, target_size):
        k_w = float(target_size) / float(source_w)
        k_h = float(target_size) / float(source_h)
        direct_w = torch.linspace(0, k_w, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
        direct_h = torch.linspace(0, k_h, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)

        full = torch.cat([direct_w, direct_w.transpose(1,0)], dim=2).unsqueeze(0)
        return full

    def random_crop_grid(x, grid):
        delta = x.size(2) - grid.size(1)
        grid = grid.repeat(x.size(0), 1, 1, 1)
        # Add random shifts by x
        grid[:, :, :, 0] = grid[:, :, :, 0] + torch.FloatTensor(x.size(0)).random_(0, delta).unsqueeze(-1).unsqueeze(-1).\
            expand(-1, grid.size(1), grid.size(2)) / x.size(2)
        # Add random shifts by y
        grid[:, :, :, 1] = grid[:, :, :, 1] + torch.FloatTensor(x.size(0)).random_(0, delta).unsqueeze(-1).unsqueeze(-1).\
            expand(-1,grid.size(1),grid.size(2)) / x.size(2)
        return grid

    #We want to crop a 80x80 image randomly for our batch
    #Building central crop of 80 pixel size
    grid = build_grid(x.shape[1], x.shape[2], target_size)
    #Make radom shift for each batch
    grid_shifted = random_crop_grid(x,grid)
    #Sample using grid sample
    sampled_batch = NN.grid_sample(x.float(), grid_shifted)

    x = get_img(grid[0], 0).astype(np.uint8)*4
    plt.imshow(x)
    plt.show()
    return sampled_batch


def train_GADA(gada: GADA,
               trainset: GADAsetFactory,
               testset: GADAsetFactory,
               opt_fn: Callable[[], torch.optim.Optimizer],
               nb_epochs=1000,
               bs=64,
               crop_dim=128,
               test_bs=None,
               test_period=50,
               data_in_gpu=False,
               log_dir='./logs/'):
    from dataset_tid import TID2013
    test_bs = bs if test_bs is None else test_bs

    if data_in_gpu:
        train_dataset = trainset.tensor_dataset(crop_dim, device=gada.device)
        # test_dataset = testset.tensor_dataset(crop_dim, device=gada.device)
    else:
        train_dataset = trainset.tensor_dataset(crop_dim, device=None)
        # test_dataset = testset.tensor_dataset(crop_dim, device=None)

    if isinstance(trainset, TID2013):
        trainset.drop_images()
    if isinstance(testset, TID2013):
        testset.drop_images()

    if data_in_gpu:
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
        # test_loader =  DataLoader(test_dataset, batch_size=test_bs, shuffle=False, num_workers=0, pin_memory=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
        # test_loader =  DataLoader(test_dataset, batch_size=test_bs, shuffle=False, num_workers=0, pin_memory=True)

    opt = torch.optim.Adam(gada.parameters(), lr=0.001, betas=(0.5, 0.999))

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    summary_writer = SummaryWriter(join(log_dir, timestamp))

    for epoch in range(1, nb_epochs+1):
        nb_batches = len(train_loader)

        #lll = []
        epoch_start_time = time.time()
        for batch, (x_ref, x_dist, y_dist_type, y_dist_level, id) in enumerate(train_loader):
            y_dist_type = index_to_1hot((y_dist_type-1)[:,0].cpu().numpy(), gada.nb_distortions).to(gada.device) # move this op in GPU to gain ~50ms per epoch

            if data_in_gpu:
                x_ref, x_dist, y_dist_level, id = to_float32(x_ref, x_dist, y_dist_level, id)
            else:
                x_ref, x_dist, y_dist_level, id = to_device(*to_float32(x_ref, x_dist, y_dist_level, id), device=gada.device)
            x_ref /= 255.
            x_dist /= 255.
            losses = gada.train_on_batch(x_ref, x_dist, y_dist_type, y_dist_level, opt)
            losses.summary_write(summary_writer, epoch*nb_batches + batch+1)  # ~200ms per epoch
            #lll.append(losses)
            print(f"Epoch[{epoch}/{nb_epochs}] Batch[{batch+1}/{nb_batches}] -- Loss = {losses.loss}")
        #ep_loss = GADA.Loss.sum_all(lll)
        print(f"----------------------------------------------------- Epoch executed in {time.time()-epoch_start_time} seconds.")
        #loss_stats = LossStats.sum_all(loss_stats)
        #loss_stats.summary_write(summary_writer, epoch, 'train--')

        #
        # if epoch%test_period == 0:
        #     loss_stats = []
        #     for batch, (x_ref, x_dist, y_dist_type, y_dist_level, id) in enumerate(test_loader):
        #         (x_ref, x_dist, y_dist_type, y_dist_level, id) = to_float32(x_ref, x_dist, y_dist_type, y_dist_level, id)
        #         _, running_loss[batch] = gada.forward_w_loss(x_ref, x_dist, y_dist_type, y_dist_level)
        #     loss_stats = GADA.LossStats.sum_all(loss_stats)
        #     loss_stats.summary_write(summary_writer, epoch, 'test--')

        #print(f'Epoch {epoch}/{nb_epochs}: avg-loss={np.mean(running_loss)}')


    @property
    def device():
        return list(gada.parameters())[0].device
