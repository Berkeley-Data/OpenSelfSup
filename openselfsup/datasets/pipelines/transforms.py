import cv2
import inspect
import numpy as np
from PIL import Image, ImageFilter
import albumentations as A

import torch
from torchvision import transforms as _transforms

from openselfsup.utils import build_from_cfg

from ..registry import PIPELINES

# register all existing transforms in torchvision
_EXCLUDED_TRANSFORMS = ['GaussianBlur']
for m in inspect.getmembers(_transforms, inspect.isclass):
    if m[0] not in _EXCLUDED_TRANSFORMS:
        PIPELINES.register_module(m[1])


@PIPELINES.register_module
class Alb_ColorJitter(object):

    def __call__(self, img):
        # TODO: Fix the code
        transform = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
        ])
        # Augment an image
        transformed = transform(image=img)
        transformed_image = transformed["image"]
        return transformed_image

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class Alb_RandomCrop(object):

    def __call__(self, img):

        # TODO: Fix the code
        transform = A.Compose([
            A.RandomCrop(width=196, height=196),
        ])
        # Augment an image
        transformed = transform(image=img)
        transformed_image = transformed["image"]
        return transformed_image

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class Alb_GaussianBlur(object):

    def __call__(self, img):

        transform = A.Compose([
            A.GaussianBlur (blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5),
        ])
        # Augment an image
        transformed = transform(image=img)
        transformed_image = transformed["image"]
        return transformed_image

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class Alb_RandomBrightnessContrast(object):

    def __call__(self, img):

        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.5),
        ])
        # Augment an image
        transformed = transform(image=img)
        transformed_image = transformed["image"]
        return transformed_image

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class Alb_ElasticTransform(object):

    def __call__(self, img):

        transform = A.Compose([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, always_apply=False, p=0.5),
        ])
        # Augment an image
        transformed = transform(image=img)
        transformed_image = transformed["image"]
        return transformed_image

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class Alb_Blur(object):

    def __call__(self, img):

        transform = A.Compose([
            A.Blur(blur_limit=7, always_apply=False, p=0.5),
        ])
        # Augment an image
        transformed = transform(image=img)
        transformed_image = transformed["image"]
        return transformed_image

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module
class Alb_VerticalFlip(object):

    def __call__(self, img):

        transform = A.Compose([
            A.VerticalFlip(p=0.5),
        ])
        # Augment an image
        transformed = transform(image=img)
        transformed_image = transformed["image"]
        return transformed_image

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module
class Alb_HorizontalFlip(object):

    def __call__(self, img):

        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
        ])
        # Augment an image
        transformed = transform(image=img)
        transformed_image = transformed["image"]
        return transformed_image

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class RandomAppliedTrans(object):
    """Randomly applied transformations.

    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float): Probability.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


# custom transforms
@PIPELINES.register_module
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)."""

    _IMAGENET_PCA = {
        'eigval':
        torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec':
        torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }

    def __init__(self):
        self.alphastd = 0.1
        self.eigval = self._IMAGENET_PCA['eigval']
        self.eigvec = self._IMAGENET_PCA['eigvec']

    def __call__(self, img):
        assert isinstance(img, torch.Tensor), \
            "Expect torch.Tensor, got {}".format(type(img))
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class Solarization(object):
    """Solarization augmentation in BYOL https://arxiv.org/abs/2006.07733."""

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        img = np.array(img)
        img = np.where(img < self.threshold, img, 255 -img)
        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module
class Sen12msNormalize(object):
    def __init__(self, bands_mean, bands_std):

        self.bands_s1_mean = bands_mean['s1_mean']
        self.bands_s1_std = bands_std['s1_std']

        self.bands_s2_mean = bands_mean['s2_mean']
        self.bands_s2_std = bands_std['s2_std']

        self.bands_RGB_mean = bands_mean['s2_mean'][0:3]
        self.bands_RGB_std = bands_std['s2_std'][0:3]

        self.bands_all_mean = self.bands_s2_mean + self.bands_s1_mean
        self.bands_all_std = self.bands_s2_std + self.bands_s1_std

    def __call__(self, rt_sample):

        # img, label, sample_id = rt_sample['image'], rt_sample['label'], rt_sample['id']
        img = rt_sample

        # different input channels
        if img.size()[0] == 12:
            for t, m, s in zip(img, self.bands_all_mean, self.bands_all_std):
                t.sub_(m).div_(s)
        elif img.size()[0] == 10:
            for t, m, s in zip(img, self.bands_s2_mean, self.bands_s2_std):
                t.sub_(m).div_(s)
        elif img.size()[0] == 5:
            for t, m, s in zip(img,
                               self.bands_RGB_mean + self.bands_s1_mean,
                               self.bands_RGB_std + self.bands_s1_std):
                t.sub_(m).div_(s)
        elif img.size()[0] == 3:
            for t, m, s in zip(img, self.bands_RGB_mean, self.bands_RGB_std):
                t.sub_(m).div_(s)
        else:
            for t, m, s in zip(img, self.bands_s1_mean, self.bands_s1_std):
                t.sub_(m).div_(s)

        # return {'image': img, 'label': label, 'id': sample_id}
        return img

@PIPELINES.register_module
class Sen12msToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, rt_sample):
        # img, label, sample_id = rt_sample['image'], rt_sample['label'], rt_sample['id']

        # rt_sample = {'image': torch.tensor(img), 'label': label, 'id': sample_id}
        return torch.tensor(rt_sample)
