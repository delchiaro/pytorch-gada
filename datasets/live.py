from pathlib import Path
from typing import Dict

import numpy as np
from datasets.gadaset import GADAsetFactory


class LIVE(GADAsetFactory):

    def _dist_type_names(self) -> np.ndarray:
        return np.array(['jp2K', 'jpeg', 'wn', 'gblur', 'fastfading'])

    def __init__(self, dataset_path, load_dataset=False):
        self.path = Path(dataset_path)
        super().__init__(load_dataset)

    def _init(self, metadata, remove_origs=True) -> dict:
        import scipy.io
        import os


        def get_fnames(noise_folder_name):
            names = [noise_folder_name + '/' + f for f in sorted(os.listdir(self.path / noise_folder_name)) if f.endswith('.bmp')]
            new_names = []
            for name in names:
                _ = name.split('img')
                name_0 = 'img'.join(_[:-1])
                name_1 = _[-1]
                number, ext = name_1.split('.')
                new_name = name_0 + 'img' + f"{int(number):03d}" + '.' + ext
                new_names.append(new_name)
            ordering = np.argsort(new_names)
            return list(np.array(names)[ordering])

        dist_im_names = np.array(get_fnames('jp2K') + get_fnames('jpeg') + get_fnames('wn') +
                                 get_fnames('gblur') + get_fnames('fastfading'))

        # dmos = [dmos_jpeg2000(1:227) dmos_jpeg(1:233) white_noise(1:174) gaussian_blur(1:174) fast_fading(1:174)]
        # where dmos_distortion(i) is the dmos value for image "distortion\imgi.bmp" where distortion can be one of the five described above
        dist_categ = np.hstack((np.zeros([227], dtype=np.uint8),
                                np.ones([233], dtype=np.uint8),
                                np.ones([174], dtype=np.uint8) * 2,
                                np.ones([174], dtype=np.uint8) * 3,
                                np.ones([174], dtype=np.uint8) * 4))
        dmos = scipy.io.loadmat(self.path / 'dmos.mat')
        dist_level = dmos['dmos'][0]/100


        # The file refnames_all.mat contains a cell array refnames_all. Entry refnames_all{i} is the name of
        # the reference image for image i whose dmos value is given by dmos(i).
        ref_im_names = np.hstack(scipy.io.loadmat(self.path / 'refnames_all.mat')['refnames_all'][0])

        # If orgs(i)==0, then this is a valid dmos entry. Else if orgs(i)==1 then image i denotes a copy of the reference image.
        # The reference images are provided in the folder refimgs.
        orgs = np.squeeze(np.argwhere(dmos['orgs'][0]==1))
        if remove_origs:
            dist_im_names = np.delete(dist_im_names, orgs, axis=0)
            ref_im_names = np.delete(ref_im_names, orgs, axis=0)
            dist_categ = np.delete(dist_categ, orgs, axis=0)
            dist_level = np.delete(dist_level, orgs, axis=0)

        metadata = {GADAsetFactory.COL_DISTORTED_NAME: dist_im_names,
                    GADAsetFactory.COL_REFERENCE_NAME: ref_im_names,
                    GADAsetFactory.COL_DISTORTION_CATEG: dist_categ,
                    GADAsetFactory.COL_DISTORTION_LEVEL: dist_level}
        return metadata

    def _image_loader(self, threads, verbose, **kwargs):
        dist_path = self.path
        ref_path = self.path/'refimgs'
        return self._default_image_loeader(ref_path, dist_path, threads, verbose)

    def __repr__(self):
        r = super().__repr__()
        return str(f"path: {self.path}\n\n{r}")







#%%
if __name__ == "__main__":

    #%%
    live = LIVE('/mnt/2tb/datasets/IQA/live/').init(remove_origs=False)
    live.load_images()
    print(live)
    splits = live.generate_ref_splits('./data/live_splits', ref_imgs_per_split=5, n_splits=10, save=True)
    splits_loaded = live.load_ref_splits('./data/live_splits')

    train, val = live.train_test_split_by_ref(n_random=5, specify='test')
    train.save('./data/live_train_random_5.pickle')
    val.save('./data/live_test_random_5.pickle')
    live.load('./data/live_train_random_5.pickle')

    #%%
    live_train = LIVE('/mnt/2tb/datasets/IQA/live/').load('./data/live_train_random_5.pickle')
    live_test = LIVE('/mnt/2tb/datasets/IQA/live/').load('./data/live_test_random_5.pickle')
    live_train.save_ref_split('./data/live_train_ref_split.split')
    live_test.save_ref_split('./data/live_test_ref_split.split')


    #%%


    #%%
    live = LIVE('/mnt/2tb/datasets/IQA/live/').load('live.pickle')
    live_train = live.load_split('live_train_2200.split')
    live_test = live.load_split('live_test_800.split')