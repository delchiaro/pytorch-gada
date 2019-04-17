import torch
from torch import optim

from dataset_tid import TID2013
import numpy as np


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
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")

    print("Torch will use device: {}".format(device))
    return device
#%%




device = exp_init(gpu_index=0, seed=42)

#tid = TID2013('/mnt/2tb/datasets/IQA/tid2013/').load('tid.pickle')
tid = TID2013('/mnt/2tb/datasets/IQA/tid2013/').init()
tid_train = tid.load_split_file('tid_train_2200.split')
tid_test = tid.load_split_file('tid_test_800.split')

from gada import GADA, train_GADA





gada = GADA(nb_distortions=24, z_noise_dim=96, device=device)
train_GADA(gada, tid_train, tid_test, opt_fn=lambda p: torch.optim.Adam(p),
           nb_epochs=10000, bs=64, crop_dim=128,
           test_period=2,
           data_in_gpu=True)