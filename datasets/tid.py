from pathlib import Path
import numpy as np
from datasets.gadaset import GADAsetFactory

#
# class ImageLoader:
#     ref_images = {}
#     dist_images = {}
#
#     def __init__(self, dist_path, ref_path):
#         self.dist_path = dist_path
#         self.ref_path = ref_path
#
#     def __call__(self, t):
#         index, dist_fname, ref_fname = t
#         dist = np.array(imread(self.dist_path / dist_fname)).swapaxes(2,1).swapaxes(1,0)
#         ref = np.array(imread(self.ref_path / ref_fname)).swapaxes(2,1).swapaxes(1,0)
#         return index, dist, ref



class TID2013(GADAsetFactory):

    def _dist_type_names(self) -> np.ndarray:
        pass # TODO

    def __init__(self, dataset_path, load_dataset=False):
        self.path = Path(dataset_path)
        super().__init__(load_dataset)

    def _init(self, metadata) -> dict:
        lines = open(self.path / 'mos_with_names.txt', 'r').read().split('\n')
        lines.remove('')
        dist_im_names = [l.split(' ')[1] for l in lines]
        dist_level = [float(l.split(' ')[0])/9 for l in lines]
        ref_im_names = [n.split('_')[0].upper() + '.BMP' for n in dist_im_names]
        dist_categ = [int(n.split('.')[0].split('_')[1])-1 for n in dist_im_names]
        #dist_level = [float(n.split('.')[0].split('_')[2])/5 for n in dist_im_names]

        metadata = {GADAsetFactory.COL_DISTORTED_NAME: dist_im_names,
                    GADAsetFactory.COL_REFERENCE_NAME: ref_im_names,
                    GADAsetFactory.COL_DISTORTION_CATEG: dist_categ,
                    GADAsetFactory.COL_DISTORTION_LEVEL: dist_level}
        return metadata
    #

    def _image_loader(self, threads, verbose, **kwargs):
        dist_path = self.path/'distorted_images'
        ref_path = self.path/'reference_images'
        return self._default_image_loeader(ref_path, dist_path, threads, verbose)

    def __repr__(self):
        r = super().__repr__()
        return str(f"path: {self.path}\n\n{r}")







#%%
if __name__ == "__main__":

    #%%
    tid = TID2013('/mnt/2tb/datasets/IQA/tid2013/').init()
    tid.load_images()
    print(tid)
    splits = tid.generate_ref_splits('./data/tid_splits', ref_imgs_per_split=5, n_splits=10, save=True)
    splits_loaded = tid.load_ref_splits('./data/tid_splits')

    train, val = tid.train_test_split_by_ref(n_random=5, specify='test')

    train.save('train.pickle')
    tid.load('train.pickle')

    #%%
    tid_train = TID2013('/mnt/2tb/datasets/IQA/tid2013/').load('tid_train_2200.pickle')
    tid_test = TID2013('/mnt/2tb/datasets/IQA/tid2013/').load('tid_test_800.pickle')
    tid_train.save_ref_split('tid_train_2200.split')
    tid_test.save_ref_split('tid_test_800.split')


    #%%


    #%%
    tid = TID2013('/mnt/2tb/datasets/IQA/tid2013/').load('tid.pickle')
    tid_train = tid.load_split('tid_train_2200.split')
    tid_test = tid.load_split('tid_test_800.split')