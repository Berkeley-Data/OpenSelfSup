from abc import ABCMeta, abstractmethod

import torch
from torch.utils.data import Dataset

from openselfsup.utils import print_log, build_from_cfg

from torchvision.transforms import Compose

from .registry import DATASETS, PIPELINES
from .builder import build_datasource


class Sen12msDataset(Dataset):
    """PyTorch dataset class for the SEN12MS dataset"""

    def __init__(self, data_source, pipeline, prefetch=False, use_s2=True, use_s1=True, use_RGB=False):
        self.data_source = build_datasource(data_source)
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline = Compose(pipeline)
        self.prefetch = prefetch

    def __getitem__(self, index):
        """Get a single example from the dataset"""
        return self.data_source.get_sample(index)

    def __len__(self):
        return self.data_source.get_length()




