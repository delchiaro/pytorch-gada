from gada.wgada import train_WGADA
from run_settings import device, log_dir, crop_dim, nb_dist,  wgada, trainset, testset, state_dir, jpeg_grid
from gada import train_GADA

print(wgada.G)
wgada.summary(crop_dim, channels=3, print_params=True)

train_WGADA(wgada, trainset, testset,  # opt_fn=lambda p: torch.optim.Adam(p),
            nb_epochs=20000, bs=16, crop_dim=crop_dim, jpeg_grid=jpeg_grid,
            # state_dir=None,
            # log_dir=None,
            state_dir=state_dir,
            log_dir=log_dir,
            first_save_state_epoch=0,
            save_state_interval=50,
            data_in_gpu=False,
            nb_critic_iter=3,
            test_period=50, )
