from copy import deepcopy
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



class WATERLOO(GADAsetFactory):

    def __init__(self, dataset_path, load_dataset=False):
        self.path = Path(dataset_path)
        super().__init__(load_dataset)

    def _init(self, metadata) -> dict:
        import os
        dist_im_names = np.array([im_name for im_name in sorted(os.listdir(self.path/'pristine_images')) if im_name.endswith('.bmp')])
        ref_im_names = deepcopy(dist_im_names)
        dist_level = np.array([.0 for im in dist_im_names])
        dist_categ = np.array([0 for im in dist_im_names])

        metadata = {WATERLOO.COL_DISTORTED_NAME: dist_im_names,
                    WATERLOO.COL_REFERENCE_NAME: ref_im_names,
                    WATERLOO.COL_DISTORTION_CATEG: dist_categ,
                    WATERLOO.COL_DISTORTION_LEVEL: dist_level}
        return metadata
    #
    # def _load_images(self, threads, verbose, **kwargs):
    #     if verbose:
    #         print(f'Loading image files from folder {self.path}  ...')
    #     lists = [self.data.index, list(self.data[TID2013.COL_DISTORTED_NAME]), list(self.data[TID2013.COL_REFERENCE_NAME])]
    #     indices = self.data.index
    #     args = []
    #     for i, d, r in zip(*lists):
    #         if i in indices:
    #             args.append((i, d, r))
    #
    #
    #     proc = ImageLoader(dist_path, ref_path)
    #     pool = mp.Pool(threads)
    #     res = pool.map(proc, args)
    #
    #     #idx = [r[0] for r in res]
    #     dist = [r[1] for r in res]
    #     ref = [r[2] for r in res]
    #     return np.array(ref, dtype=np.uint8), np.array(dist, dtype=np.uint8)

    def _image_loader(self, threads, verbose, **kwargs):
        dist_path = self.path/'pristine_images'
        ref_path = self.path/'pristine_images'
        return self._default_image_loeader(ref_path, dist_path, threads, verbose)

    def __repr__(self):
        r = super().__repr__()
        return str(f"path: {self.path}\n\n{r}")







#%%
if __name__ == "__main__":

    #%%
    waterloo = WATERLOO('/mnt/2tb/datasets/IQA/WATERLOO/').init()
    waterloo.load_images(verbose=True)
    print(waterloo)
    #splits = waterloo.generate_ref_splits('./data/waterloo_split', ref_imgs_per_split=5, n_splits=10, save=True)
    #splits_loaded = waterloo.load_ref_splits('./data/waterloo_split')

    #train, val = waterloo.train_test_split_by_ref(n_random=5, specify='test')

    #train.save('train.pickle')
   # waterloo.load('train.pickle')

    #%%


    #%%


    #%%
