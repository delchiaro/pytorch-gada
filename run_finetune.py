from dataset_waterloo import WATERLOO
from gada.gada import finetune_GADA
from run_settings import device, log_dir, crop_dim, nb_dist, gada, trainset, testset, state_dir, jpeg_grid,  state_timestamp_test, state_epoch_test
from gada import train_GADA

print(gada.G)
gada.summary(crop_dim, channels=3, print_params=True)


import torch
state_timestamp_test= '2019-05-16_18:20:34'
state_epoch_test = 20000
gada.load_state_dict(torch.load(state_dir+f'/{state_timestamp_test}/gada-[ep={state_epoch_test}]', map_location=gada.device))


lr = 1e-5
for param_group in gada.opt_d.param_groups:
    param_group['lr'] = lr


ft_dataset =  WATERLOO('/mnt/2tb/datasets/IQA/WATERLOO/').init()
ft_dataset.load_images(verbose=True)
#dataset.generate_ref_splits('./data/live_splits', ref_imgs_per_split=5, n_splits=10, save=True)


finetune_GADA(gada, ft_dataset, testset, #opt_fn=lambda p: torch.optim.Adam(p),
              gen_dist_from_trainset=True,
              use_dist_from_trainset=False,
               nb_epochs=1000, bs=32, crop_dim=crop_dim, jpeg_grid=jpeg_grid,
               #state_dir=None,
               #log_dir=None,
               state_dir=None,#ft_state_dir,
               log_dir=None, #ft_log_dir,
               first_save_state_epoch=0,
               save_state_interval=50,
               data_in_gpu=False,
               test_period=10,
              train_classifier=False)
