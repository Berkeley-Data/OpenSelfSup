from ..registry import DATASOURCES
from .utils import McLoader

import os
import random
import rasterio
import numpy as np
from tqdm import tqdm

# indices of sentinel-2 bands related to land
S2_BANDS_LD = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]
S2_BANDS_RGB = [2, 3, 4] # B(2),G(3),R(4)  [todo] temp fix for running code end to end.

# util function for reading s2 data
def load_s2(path, imgTransform, s2_band):
    bands_selected = s2_band
    with rasterio.open(path) as data:
        s2 = data.read(bands_selected)
    s2 = s2.astype(np.float32)
    if not imgTransform:
        s2 = np.clip(s2, 0, 10000)
        s2 /= 10000
    s2 = s2.astype(np.float32)
    return s2

# util function for reading s1 data
def load_s1(path, imgTransform):
    with rasterio.open(path) as data:
        band1 = data.read(1)
        band2 = data.read(2)

    band1 = band1.astype(np.float32)
    band1 = np.nan_to_num(band1)
    band1 = np.clip(band1, -25, 0)
    if not imgTransform:
        band1 /= 25
        band1 += 1

    band2 = band2.astype(np.float32)
    band2 = np.nan_to_num(band2)
    band2 = np.clip(band2, -25, 0)
    if not imgTransform:
        band2 /= 25
        band2 += 1

    # band3 = abs(band2 - band1)
    # band3 /= 25

    s1 = np.stack((band1, band2)) # , band3))
    return s1


# util function for reading data from single sample
def load_sample(sample, imgTransform, use_s1, use_s2, use_RGB):
    # load s2 data

    img_s1 = None
    img_s2 = None

    if use_s2:
        img_s2 = load_s2(sample["s2"], imgTransform, s2_band=S2_BANDS_LD)
    # load only RGB
    elif use_RGB and use_s2 == False:
        img_s2 = load_s2(sample["s2"], imgTransform, s2_band=S2_BANDS_RGB)

    # load s1 data
    if use_s1:
        img_s1 = load_s1(sample["s1"], imgTransform)

    # Colorado comment out for now
    # t = _transforms.ToPILImage()
    # img_s1 = t(img_s1)
    # img_s2 = t(img_s2)
    return img_s1, img_s2


#  calculate number of input channels
def get_ninputs(use_s1, use_s2, use_RGB):
    n_inputs = 0
    if use_s2:
        n_inputs += len(S2_BANDS_LD)
    if use_s1:
        n_inputs += 2
    if use_RGB and use_s2 == False:
        n_inputs += 3

    return n_inputs


@DATASOURCES.register_module
class Sen12MSImageList(object):

    def __init__(self, root, list_file, memcached=False, mclient_path=None, return_label=True):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.has_labels = len(lines[0].split()) == 2
        self.return_label = return_label
        if self.has_labels:
            self.fns, self.labels = zip(*[l.strip().split() for l in lines])
            self.labels = [int(l) for l in self.labels]
        else:
            # assert self.return_label is False
            self.fns = [l.strip() for l in lines]
        self.fns = [os.path.join(root, fn) for fn in self.fns]
        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1=True, use_s2=True, use_RGB=False)

        pbar = tqdm(total=len(self.fns))  # 18106 samples in test set
        pbar.set_description("[Load]")

        # remove broken file
        broken_file = 'ROIs1868_summer_s2_146_p202.tif'
        if broken_file in self.fns:
            self.fns.remove(broken_file)

        pbar.set_description("[Load]")

        # [todo:tg] temp hard code
        self.samples = []

        path = "data/sen12ms/data"
        for s2_id in self.fns:
            s2_id = os.path.basename(s2_id)
            mini_name = s2_id.split("_")
            s2_loc = os.path.join(path, (mini_name[0] + '_' + mini_name[1]),
                                  (mini_name[2] + '_' + mini_name[3]), s2_id)
            s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")

            pbar.update()
            self.samples.append({"s1": s1_loc, "s2": s2_loc,
                                 "id": s2_id})

        pbar.close()

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])
        print(f"loaded {len(self.samples)} from {path}")

        # Commented the following as we don't need images at this stage. If we want, we can enable during debugging.
        # try:
        #     rand_index = random.randint(0, len(self.samples)-1)
        #     sample_data = self.samples[rand_index]
        #     images_title = 'Sample images during loading'
        #     wandb_utils.add_images_to_wandb(sample_data["s1"], sample_data["s2"],title=images_title, is_pixel_data=False)
        #     print(f"Added {images_title} from index {rand_index} to wandb")
        #
        # except Exception as e:
        #     print(e)

        self.fns = self.samples



    def _init_memcached(self):
        if not self.initialized:
            assert self.mclient_path is not None
            self.mc_loader = McLoader(self.mclient_path)
            self.initialized = True

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        if self.memcached:
            self._init_memcached()
        if self.memcached:
            img = self.mc_loader(self.fns[idx])
            return img
        else:
            img_s1, img_s2 = load_sample(self.fns[idx], None, use_s1=True, use_s2=True, use_RGB=False)
            return img_s1, img_s2
