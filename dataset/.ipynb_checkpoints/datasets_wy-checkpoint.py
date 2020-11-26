import os
import numpy as np
import random
import glob
import torch
import cv2
import json
from torch.utils import data
from utils.transforms import get_affine_transform
from PIL import Image


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def read_mask(mask_path): # Changed by Xinchen Liu
    """Keep reading mask until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(mask_path):
        raise IOError("{} does not exist".format(mask_path))
    while not got_img:
        try:
            mask = Image.open(mask_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(mask_path))
            pass
    return mask

        
class WYDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=[384, 384], scale_factor=0.,
                 rotation_factor=90, ignore_label=255, transform=None, list_path=None):

        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.transform = transform[0]
        self.transform_anno = transform[1]
        self.dataset = dataset
        
        if 'train' in self.dataset:
            self.list_file = 'train_list.txt'
        elif 'val' in self.dataset:
            self.list_file = 'val_list.txt'
        elif 'test_no_label' in self.dataset:
            self.list_file = 'test_no_label_list.txt'
            
        self.img_list = open(os.path.join(self.root, self.list_file)).readlines()
        self.img_list = [x.strip() for x in self.img_list]
        self.number_samples = len(self.img_list)
        

    def __len__(self):
        return self.number_samples


    def __getitem__(self, index):
        # Load training image

        im_path = os.path.join(self.root, 'images_all', self.img_list[index])
        anno_path = os.path.join(self.root, 'annotations_all', self.img_list[index].replace('jpg', 'png'))

        im = read_image(im_path)
        is_rotated = 0
        
        if os.path.exists(anno_path):
            parsing_anno = read_mask(anno_path)
            
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            random.seed(seed)  # apply this seed to img transforms
            if self.transform is not None:
                im = self.transform(im)

            random.seed(seed)  # apply this seed to mask transforms
            if self.transform_anno is not None:
                parsing_anno = self.transform_anno(parsing_anno)
           
            is_ndarray = isinstance(im, np.ndarray)
            if not is_ndarray:
                im = np.asarray(im, dtype=np.uint8)
                parsing_anno = np.asarray(parsing_anno, dtype=np.uint8)
            if im.shape[1] > im.shape[0]:
                im = im.transpose(1,0,2)
                parsing_anno = parsing_anno.transpose(1,0)
                is_rotated = 1

        else:
            im = self.transform(im)
            
            is_ndarray = isinstance(im, np.ndarray)
            if not is_ndarray:
                im = np.asarray(im, dtype=np.uint8)
            if im.shape[1] > im.shape[0]:
                im = im.transpose(1,0,2)
                is_rotated = 1

        if 'train' in self.dataset and self.crop_size[0] > 1024:
            y_s = np.random.randint(2788-self.crop_size[0])
            x_s = np.random.randint(1400-self.crop_size[1])
            im = im[y_s:y_s+self.crop_size[0], x_s:x_s+self.crop_size[1]]
            parsing_anno = parsing_anno[y_s:y_s+self.crop_size[0], x_s:x_s+self.crop_size[1]]
#             print(im.shape)
#             print(parsing_anno.shape)
            
        if 'test_no_label' in self.dataset:
            return im, is_rotated
        else:
            return im, parsing_anno, is_rotated
        