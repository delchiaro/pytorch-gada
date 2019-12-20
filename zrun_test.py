from zrun_settings import device, log_dir, crop_dim, nb_dist, wgada, trainset, testset, state_dir, state_timestamp_test, state_epoch_test, jpeg_grid
from gada import plot_GADA_gen_imgs, test_GADA
from gada.wgada import plot_WGADA_gen_imgs

#test_GADA(gada, testset, bs=128, crop_dim=128, data_in_gpu=False,  verbose=True,

plot_WGADA_gen_imgs(wgada, testset, bs=32, crop_dim=crop_dim, jpeg_grid=jpeg_grid,
                   data_in_gpu=False,  verbose=True,
                   #state_path='/mnt/2tb/datasets/IQA/live_gada_weights/gada-[ep=150]',
                   #state_path=state_dir+f'/{state_timestamp_test}/gada-[ep={state_epoch_test}]',
                   state_path=state_dir+f'/{state_timestamp_test}/gada-[ep={state_epoch_test}]',
                   #state_path='/mnt/2tb/datasets/IQA/tid2013_gada_weights/gada-[ep=150]'
                   #state_path='/mnt/2tb/datasets/IQA/live_gada_pytorch_weights/gada-[ep=400]'
                   skip_batches=0,
                   )


