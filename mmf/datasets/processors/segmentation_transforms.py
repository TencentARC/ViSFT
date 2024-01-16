# Copyright (c) Facebook, Inc. and its affiliates.

# Mostly copy-pasted from
# https://github.com/facebookresearch/detr/blob/master/datasets/transforms.py
import random
from typing import List, Optional, Union

import torch
import math
import PIL
import torchvision.transforms as T
import torchvision.transforms.functional as F
from mmf.common.registry import registry
from mmf.datasets.processors.processors import BaseProcessor
from mmf.utils.box_ops import box_xyxy_to_cxcywh
from torch import Tensor
from mmf.utils.misc import interpolate
def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")


    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target

def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target

def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                # size = int(round(max_size * min_original_size / max_original_size))
                size = int(max_size * min_original_size // max_original_size)

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(target['masks'][:, None], size, mode="bilinear")[:, 0]

    return rescaled_image, target

def pad(image, target, padding, pad_value=0):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]), fill=pad_value)
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target

def pad_around(image, target, padding, pad_value=0):
    # we pad around the image: left, top, right, bottom
    assert len(padding) == 4

    padded_image = F.pad(image, (padding[0], padding[1], padding[2], padding[3]), fill=pad_value)
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])

    if "boxes" in target:
        # the box is in form [x, y, x, y]
        target['boxes'] = target['boxes'] + target['boxes'].new_tensor([[padding[0], padding[1], padding[0], padding[1]]])

    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (padding[0], padding[2], padding[1], padding[3]))
    return padded_image, target

@registry.register_processor("segmentation_random_size_crop")
class RandomSizeCrop(BaseProcessor):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)

@registry.register_processor("segmentation_random_rescale")
class RandomRescale(BaseProcessor):
    def __init__(self, scales, target_w, target_h):
        assert isinstance(scales, (list, tuple))
        self.scales = scales
        self.target_w = target_w
        self.target_h = target_h

    def __call__(self, img, target=None):
        input_w, input_h = img.size

        random_scale = random.uniform(self.scales[0], self.scales[1])
        target_scaled_w = random_scale * self.target_w
        target_scaled_h = random_scale * self.target_h


        output_scale = min(target_scaled_w / input_w, target_scaled_h / input_h)

        output_w = round(output_scale * input_w)
        output_h = round(output_scale * input_h)

        return resize(img, target, (output_w, output_h))

@registry.register_processor("segmentation_random_resize")
class RandomResize(BaseProcessor):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)

@registry.register_processor("segmentation_random_horizontal_flip")
class RandomHorizontalFlip(BaseProcessor):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target

@registry.register_processor("segmentation_normalize")
class Normalize(BaseProcessor):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target

@registry.register_processor("segmentation_fixed_size_crop")
class FixedSizeCrop(BaseProcessor):
    """
    If `crop_size` is smaller than the input image size, then it uses a random crop of
    the crop size. If `crop_size` is larger than the input image size, then it pads
    the right and the bottom of the image to the crop size.
    """
    def __init__(self, target_w, target_h, pad=True, pad_value=0, rand_place=False, center_pad=False):
        self.target_w = target_w
        self.target_h = target_h
        self.pad = pad
        self.pad_value = pad_value
        self.rand_place = rand_place
        self.center_pad = center_pad
        assert (self.rand_place and self.center_pad) == False, 'cannot use them simultaneously'

    def __call__(self, img, target=None):

        image_width, image_height = img.size

        crop_width = self.target_w
        crop_height = self.target_h

        if self.pad:
            pad_width = max(crop_width - image_width, 0)
            pad_height = max(crop_height - image_height, 0)
        else:
            pad_width = 0
            pad_height = 0
            crop_width = min(crop_width, image_width)
            crop_height = min(crop_height, image_height)

        if (pad_width > 0) or (pad_height > 0):
            if not self.rand_place and not self.center_pad:
                img, target = pad(img, target, (pad_width, pad_height), pad_value=self.pad_value)
            elif self.center_pad:
                pad_left = pad_width // 2
                pad_top = pad_height // 2
                pad_right = pad_width - pad_left
                pad_down = pad_height - pad_top
                img, target = pad_around(img, target, (pad_left, pad_top, pad_right, pad_down), pad_value=self.pad_value)
            else:
                pad_left = torch.randint(0, pad_width + 1, size=(1,)).item()
                pad_top = torch.randint(0, pad_height + 1, size=(1,)).item()
                pad_right = pad_width - pad_left
                pad_down = pad_height - pad_top
                img, target = pad_around(img, target, (pad_left, pad_top, pad_right, pad_down), pad_value=self.pad_value)

        region = T.RandomCrop.get_params(img, [crop_height, crop_width])
        return crop(img, target, region)

@registry.register_processor("segmentation_to_tensor")
class ToTensor(BaseProcessor):
    def __call__(self, img, target=None):
        return F.to_tensor(img), target

@registry.register_processor("segmentation_pad_to_align")
class PadtoAlign(BaseProcessor):
    def __init__(self, align):
        self.align = float(align)

    def __call__(self, img, target):
        w, h = img.size
        pad_w = int(math.ceil(w / self.align) * self.align) - w
        pad_h = int(math.ceil(h / self.align) * self.align) - h
        pad_left = 0  # pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = 0  # pad_h // 2
        pad_bottom = pad_h - pad_top
        return pad_around(img, target, [pad_left, pad_top, pad_right, pad_bottom], pad_value=128) # (124, 116, 104))

@registry.register_processor("segmentation_compose")
class Compose(BaseProcessor):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string