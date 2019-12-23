from zrun_settings import log_dir, crop_dim, nb_dist, trainset, testset, state_dir, jpeg_grid, exp_init
from zrun_settings import gada # wgada
from gada import plot_GADA_gen_imgs, test_GADA
#from wgada import plot_WGADA_gen_imgs

#test_GADA(gada, testset, bs=128, crop_dim=128, data_in_gpu=False,  verbose=True,


device = exp_init(gpu_index=1, seed=42)
gada.to(device)


state_timestamp_test = '2019-12-23_17:46:26'
state_epoch_test = 500



plot_GADA_gen_imgs(gada, testset, bs=32, crop_dim=crop_dim, jpeg_grid=jpeg_grid,
                   data_in_gpu=False,  verbose=True,
                   #state_path='/mnt/2tb/datasets/IQA/live_gada_weights/gada-[ep=150]',
                   #state_path=state_dir+f'/{state_timestamp_test}/gada-[ep={state_epoch_test}]',
                   state_path=state_dir+f'/{state_timestamp_test}/gada-[ep={state_epoch_test}]',
                   #state_path='/mnt/2tb/datasets/IQA/tid2013_gada_weights/gada-[ep=150]'
                   #state_path='/mnt/2tb/datasets/IQA/live_gada_pytorch_weights/gada-[ep=400]'
                   skip_batches=0,
                   )


