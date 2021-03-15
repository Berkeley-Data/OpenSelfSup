import torch
from PIL import Image
from .registry import DATASETS
from .base import BaseDataset
from .utils import to_numpy
from .sen12ms_dataset import Sen12msDataset
from openselfsup.datasets import wandb_utils

initial_images_sent_to_wandb = False
pipeline_images_sent_to_wandb = False

@DATASETS.register_module
class ContrastiveDataset(BaseDataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        data_source['return_label'] = False
        super(ContrastiveDataset, self).__init__(data_source, pipeline, prefetch)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        img1 = self.pipeline(img)
        img2 = self.pipeline(img)
        if self.prefetch:
            img1 = torch.from_numpy(to_numpy(img1))
            img2 = torch.from_numpy(to_numpy(img2))
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented


@DATASETS.register_module
class ContrastiveMSDataset(Sen12msDataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        data_source['return_label'] = False
        super(ContrastiveMSDataset, self).__init__(data_source, pipeline, prefetch)

    def __getitem__(self, idx):
        s1_img, s2_img = self.data_source.get_sample(idx)
        # assert isinstance(img1, Image.Image), \
        #     'The output from the data source must be an Image, got: {}. \
        #     Please ensure that the list file does not contain labels.'.format(
        #     type(img1))

        # Commented the following as we don't need images before the pipeline. If we want, we can enable during debugging.
        # global initial_images_sent_to_wandb
        # try:
        #     if initial_images_sent_to_wandb == False:
        #         images_title = 'Images before pipeline'
        #         wandb_utils.add_images_to_wandb(s1_img, s2_img, title=images_title)
        #         initial_images_sent_to_wandb = True
        #         print(f"Added {images_title}")
        # except Exception as e: print(e)

        s1_img = self.pipeline(s1_img)
        s2_img = self.pipeline(s2_img)

        global pipeline_images_sent_to_wandb
        try:
            if pipeline_images_sent_to_wandb == False:
                images_title = 'Images after pipeline'
                wandb_utils.add_images_to_wandb(s1_img, s2_img, title=images_title)
                pipeline_images_sent_to_wandb = True
                print(f"Added {images_title}")
        except Exception as e: print(e)


        # Colorado: I would suggest to not mess with prefetch -- it's unstable with OpenSelfSup
        # if self.prefetch:
        #     s1_img = torch.from_numpy(to_numpy(s1_img))
        #     s2_img = torch.from_numpy(to_numpy(s2_img))
        # img_cat = torch.cat((s1_img.unsqueeze(0), s2_img.unsqueeze(0)), dim=0)

        return dict(img_k=s1_img, img_q=s2_img)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented
