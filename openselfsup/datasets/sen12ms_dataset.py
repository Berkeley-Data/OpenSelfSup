from abc import ABCMeta, abstractmethod

import torch
from torch.utils.data import Dataset

from openselfsup.utils import print_log, build_from_cfg

from torchvision.transforms import Compose

from .registry import DATASETS, PIPELINES
from .builder import build_datasource


import os
from os import walk
import glob
import random
import rasterio
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# indices of sentinel-2 bands related to land
S2_BANDS_LD = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]
S2_BANDS_RGB = [2, 3, 4] # B(2),G(3),R(4)


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
        s1 = data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    if not imgTransform:
        s1 /= 25
        s1 += 1
    s1 = s1.astype(np.float32)
    return s1


# util function for reading data from single sample
def load_sample(sample, imgTransform, use_s1, use_s2, use_RGB):
    # load s2 data
    if use_s2:
        img = load_s2(sample["s2"], imgTransform, s2_band=S2_BANDS_LD)
    # load only RGB
    if use_RGB and use_s2 == False:
        img = load_s2(sample["s2"], imgTransform, s2_band=S2_BANDS_RGB)

    # load s1 data
    if use_s1:
        if use_s2 or use_RGB:
            img = np.concatenate((img, load_s1(sample["s1"], imgTransform)), axis=0)
        else:
            img = load_s1(sample["s1"], imgTransform)

    # load label
    # lc = labels[sample["id"]]

    # covert label to IGBP simplified scheme
    # if IGBP_s:
    #     cls1 = sum(lc[0:5]);
    #     cls2 = sum(lc[5:7]);
    #     cls3 = sum(lc[7:9]);
    #     cls6 = lc[11] + lc[13];
    #     lc = np.asarray([cls1, cls2, cls3, lc[9], lc[10], cls6, lc[12], lc[14], lc[15], lc[16]])

    # if label_type == "multi_label":
    #     lc_hot = (lc >= threshold).astype(np.float32)
    # else:
    #     loc = np.argmax(lc, axis=-1)
    #     lc_hot = np.zeros_like(lc).astype(np.float32)
    #     lc_hot[loc] = 1

    # rt_sample = {'image': img, 'label': lc_hot, 'id': sample["id"]}
    rt_sample = {'image': img, 'label': 'todo_summy', 'id': sample["id"]}

    if imgTransform is not None:
        rt_sample = imgTransform(rt_sample)

    return rt_sample


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

class Sen12msDataset(Dataset):
    """PyTorch dataset class for the SEN12MS dataset"""

    # expects dataset dir as:
    #       - SEN12MS_holdOutScenes.txt
    #       - ROIsxxxx_y
    #           - lc_n
    #           - s1_n
    #           - s2_n
    #
    # SEN12SEN12MS_holdOutScenes.txt contains the subdirs for the official
    # train/val/test split and can be obtained from:
    # https://github.com/MSchmitt1984/SEN12MS/

    def __init__(self, path=None, data_index_dir=None, imgTransform=None, use_s2=True, use_s1=False, use_RGB=False):
        """Initialize the dataset"""

        # inizialize
        super(Sen12msDataset, self).__init__()
        self.imgTransform = imgTransform
        # self.threshold = threshold
        # self.label_type = label_type

        # make sure input parameters are okay
        if not (use_s2 or use_s1 or use_RGB):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2, s1, RGB] to True!")
        self.use_s2 = use_s2
        self.use_s1 = use_s1
        self.use_RGB = use_RGB
        # self.IGBP_s = IGBP_s

        # assert subset in ["train", "val", "test"]
        # assert label_type in ["multi_label", "single_label"] # new !!

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2, use_RGB)

        assert os.path.exists(path)
        self.samples = []

        file = os.path.join(data_index_dir, 'small_sample.pkl')
        sample_list = pkl.load(open(file, "rb"))

        pbar = tqdm(total=len(sample_list))  # 18106 samples in test set
        pbar.set_description("[Load]")

        # remove broken file
        broken_file = 'ROIs1868_summer_s2_146_p202.tif'
        if broken_file in sample_list:
            sample_list.remove(broken_file)

        #
        pbar.set_description("[Load]")

        for s2_id in sample_list:
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

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        # labels = self.labels
        return load_sample(sample, self.imgTransform,
                           self.use_s1, self.use_s2, self.use_RGB)

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


