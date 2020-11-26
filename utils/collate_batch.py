 # encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import numpy as np
import cv2

def fast_collate_fn(batch):
    imgs, is_rotated = zip(*batch)
    is_ndarray = isinstance(imgs[0], np.ndarray)
    if not is_ndarray:  # PIL Image object
        w = imgs[0].size[0]
        h = imgs[0].size[1]
    else:
        w = imgs[0].shape[1]
        h = imgs[0].shape[0]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        if not is_ndarray:
            img = np.asarray(img, dtype=np.uint8)
        numpy_array = np.rollaxis(img, 2)
        tensor[i] += torch.from_numpy(numpy_array)
    return tensor, torch.tensor(is_rotated).long()


def fast_collate_fn_mask(batch): # Changed by Xinchen Liu
    imgs, masks, is_rotated = zip(*batch)
    is_ndarray = isinstance(imgs[0], np.ndarray)
    if not is_ndarray:  # PIL Image object
        w = imgs[0].size[0]
        h = imgs[0].size[1]
    else:
        w = imgs[0].shape[1]
        h = imgs[0].shape[0]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    tensor_mask = torch.zeros((len(imgs), 1, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        mask = masks[i]
        if not is_ndarray:
            img = np.asarray(img, dtype=np.uint8)
            mask = np.asarray(mask, dtype=np.uint8)
        numpy_array = np.rollaxis(img, 2)
        mask_array = mask[np.newaxis, :, :]
        tensor[i] += torch.from_numpy(numpy_array)
        tensor_mask[i] += torch.from_numpy(mask_array)
    return tensor, tensor_mask, torch.tensor(is_rotated).long()
