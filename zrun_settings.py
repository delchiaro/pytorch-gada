import torch

from datasets.live import LIVE
from datasets.tid import TID2013
import numpy as np

from gada import GADA
from wgada import WGADA, WGANMode


def exp_init(gpu_index=0, seed=None, use_fixed_seed=True, use_gpu=True):
    if use_fixed_seed and seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_gpu and gpu_index is not None:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    device = "cpu"
    if use_gpu and gpu_index is not None:
        torch.cuda.set_device(gpu_index)
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")

    print("Torch will use device: {}".format(device))
    return device


# %%


DATASET = 'live'

# dataset = TID2013('/mnt/2tb/datasets/IQA/tid2013/').init()
# dataset.generate_ref_splits('./data/tid_splits', ref_imgs_per_split=5, n_splits=10, save=True)
# splits = dataset.load_ref_splits('./data/tid_splits')

if DATASET is 'live':
    dataset = LIVE('/mnt/2tb/datasets/IQA/live/').init(remove_origs=True)
    # dataset.generate_ref_splits('./data/live_splits', ref_imgs_per_split=5, n_splits=10, save=True)
    splits = dataset.load_ref_splits('./data/live_splits')
    state_dir = '/mnt/2tb/datasets/IQA/live_gada_pytorch_weights'
    log_dir = './logs/live_pytorch/'
    nb_dist = 5

elif DATASET is 'tid2013':
    dataset = TID2013('/mnt/2tb/datasets/IQA/tid2013/').init()
    # dataset.generate_ref_splits('./data/tid_splits', ref_imgs_per_split=5, n_splits=10, save=True)
    splits = dataset.load_ref_splits('./data/tid_splits')
    state_dir = '/mnt/2tb/datasets/IQA/tid2013_gada_weights'
    log_dir = './logs/tid/'
    nb_dist = 24

trainset = splits[1]['train'].load_images(verbose=True)
testset = splits[1]['test'].load_images(verbose=True)

jpeg_grid = True
crop_dim = 128

GPU = 0
device = exp_init(gpu_index=GPU, seed=42)
gada = GADA(nb_distortions=nb_dist,
            z_noise_dim=96,
            # g_blocks=(64, 128, 128),                   # 2019-05-16_11:46:39
            g_blocks=((64, 1), 128, 128, 256, 256),  # 2019-05-16_18:20:34
            # g_blocks=(64, 128, 256, 256),               # 2019-05-16_18:22:37
            # g_blocks=((64,1), (64,2), (128, 2), (256, 1), (256, 2)),
            target_im_size=crop_dim,
            g_disable_skip_connections=False,
            device=device)
#
# wgada = WGADA(nb_distortions=nb_dist,
#               z_noise_dim=96,
#               lr=.0004,
#               g_blocks=(64, 128, 128),                   # 2019-05-16_11:46:39
#               #g_blocks=(64, 128, 128, 256, 256),  # 2019-05-16_18:20:34
#               # g_blocks=(64, 128, 256, 256),               # 2019-05-16_18:22:37
#               # g_blocks=((64,1), (64,2), (128, 2), (256, 1), (256, 2)),
#               target_im_size=crop_dim,
#               g_disable_skip_connections=False,
#               grad_penalty_lambda=1,
#               wgan_mode=WGANMode.penalty,
#               use_resize_conv=False,
#               device=device)

state_timestamp_test = '2019-05-16_18:20:34'
state_epoch_test = 9000

state_timestamp_test = '2019-09-12_17:25:30'
state_epoch_test = 16000


